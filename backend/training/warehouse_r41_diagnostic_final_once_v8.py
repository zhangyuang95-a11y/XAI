"""Irrevocable one-shot controller for the r4.1 diagnostic v8 final audit.

The controller authenticates the frozen development candidate without running
it, derives an output-independent attempt key, and permanently reserves that
key before the fresh holdout builder or explanation program is invoked.  The
reservation lives outside the requested output directory, so deleting partial
output cannot enable a retry.  A normal failure is recorded as a burned failed
attempt; process-level interruptions are recorded where possible and then
re-raised.

This module must only be run after development RCPD v8 passes.  Importing it,
running its tests, or reading an existing completion never accesses final data.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import pwd
import re
from typing import Any, Callable, Mapping, Sequence

from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training import warehouse_r41_diagnostic_fresh_final_holdout_v4 as holdout_api
from backend.training import warehouse_r41_diagnostic_explanation_audit_v8 as audit_api
from backend.training import warehouse_r41_diagnostic_rcpd_v8 as rcpd_api
from backend.training import warehouse_r41_diagnostic_designation_v2_binding as designation_binding
from backend.training import warehouse_r41_diagnostic_designation_v2 as designation_api
from backend.training import warehouse_r41_diagnostic_frozen_manifest_v2 as manifest_binding
from backend.training import warehouse_r41_diagnostic_input_snapshot_v8 as input_snapshot_api


VERSION = "warehouse-r41-diagnostic-final-once.v8"
ROOT = Path(__file__).resolve().parents[2]
LEDGER_ENV = "WAREHOUSE_R41_FINAL_LEDGER_DIR"
PERMANENT_ANCHOR_ENV = "WAREHOUSE_R41_FINAL_PERMANENT_ANCHOR"
SALT_FILE_ENV = "WAREHOUSE_R41_FINAL_HOLDOUT_SALT_FILE"
REGISTRY_ROOT = holdout_api.DEFAULT_LEDGER_ROOT
PERMANENT_ANCHOR = holdout_api.DEFAULT_PERMANENT_ANCHOR
_ACCOUNT_HOME = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
DEFAULT_SALT_FILE = (
    _ACCOUNT_HOME / ".config/policylens/warehouse_r41_diagnostic_v8_holdout_salt.bin"
)
MAX_JSON_BYTES = 512 * 1024 * 1024
MAX_INPUT_BYTES = 2 * 1024 * 1024 * 1024
_HEX = re.compile(r"[0-9a-f]{64}\Z")
PHASE_RECEIPT_NAMES = (
    "candidate_authenticated.json", "holdout_started.json",
    "historical_exclusion_started.json",
    "historical_exclusion_completed.json", "holdout_completed.json",
    "audit_started.json", "audit_completed.json",
)
_SUCCESS_OUTPUT_ARTIFACT_NAMES = frozenset({
    "attempt_started.json",
    "fresh_holdout/v3_exclusion.json",
    "fresh_holdout/holdout.json",
    "fresh_holdout/report.json",
    "explanation_audit/inputs.json",
    "explanation_audit/evidence.npz",
    "explanation_audit/report.json",
    "physical_replay.json",
})
_COMPLETION_FIELDS = frozenset({
    "version", "key", "campaign_key", "candidate_identity_sha256",
    "status", "identity", "producer_sources", "attempt_started_sha256",
    "permanent_anchor_sha256", "phase_receipts",
    "uncommitted_phase_residues", "completed_at", "automatic_retry",
    "retry_allowed", "reason", "output_created", "output_identity",
    "candidate_authentication_status", "candidate_authenticated_sha256",
    "candidate_artifacts", "candidate_artifacts_sha256", "holdout_status",
    "audit_status", "physical_replay_status", "artifacts", "program_fits",
    "development_authentication_refit", "actor_updates",
    "runtime_action_override", "formal_ready",
})
_PHASE_FIELDS = {
    "candidate_authenticated.json": frozenset({
        "version", "status", "campaign_key", "candidate_identity_sha256",
        "attempt_started_sha256", "candidate_artifacts",
        "candidate_artifacts_sha256", "rcpd_report_file_sha256",
        "program_file_sha256", "rows_file_sha256",
        "prior_rows_reauthentication_report_file_sha256",
        "prior_v7_source_report_file_sha256", "prior_v7_rows_file_sha256",
        "expansion_rows_reauthentication_report_file_sha256",
        "expansion_source_collection_report_file_sha256",
        "expansion_rows_file_sha256",
        "development_expansion_registry_file_sha256",
        "development_expansion_report_file_sha256", "fit_config_file_sha256",
        "actor_file_sha256", "protocol_file_sha256", "manifest_file_sha256",
        "designation_file_sha256", "selected_scenes_file_sha256",
        "development_registries", "require_passed", "refit", "formal_ready",
    }),
    "holdout_started.json": frozenset({
        "version", "status", "campaign_key", "candidate_identity_sha256",
        "attempt_started_sha256", "candidate_authenticated_sha256",
        "selection_salt_commitment",
    }),
    "historical_exclusion_started.json": frozenset({
        "version", "status", "campaign_key", "candidate_identity_sha256",
        "attempt_started_sha256", "candidate_authenticated_sha256",
        "holdout_started_sha256", "manifest_file_sha256",
        "row_artifact_evidence", "row_artifact_evidence_sha256",
        "public_exclusion_commitment", "public_exclusion_commitment_sha256",
        "historical_final_access_refunds_attempt", "formal_ready",
    }),
    "historical_exclusion_completed.json": frozenset({
        "version", "status", "campaign_key", "candidate_identity_sha256",
        "attempt_started_sha256", "candidate_authenticated_sha256",
        "holdout_started_sha256", "historical_exclusion_started_sha256",
        "manifest_file_sha256", "historical_final_identity_sha256",
        "historical_final_scene_count",
        "historical_final_public_observation_count",
        "historical_final_public_observations_sha256",
        "combined_excluded_scene_fingerprint_count",
        "combined_excluded_scene_fingerprints_sha256",
        "combined_excluded_seed_count", "combined_excluded_seeds_sha256",
        "combined_forbidden_public_observation_count",
        "combined_forbidden_public_observations_sha256",
        "combined_replayed_excluded_public_observation_count",
        "combined_replayed_excluded_public_observations_sha256",
        "development_row_scene_overlap", "preexisting_scene_identity_overlap",
        "preexisting_public_observation_overlap", "row_artifact_evidence",
        "row_artifact_evidence_sha256",
        "historical_final_replayed_for_exclusion_only",
        "historical_final_actor_executed_for_observation_exclusion_only",
        "historical_final_actor_outputs_exposed",
        "historical_final_actor_outputs_persisted", "historical_final_labels_used",
        "historical_final_saved_metrics_replayed_for_authentication",
        "historical_final_metrics_exposed", "historical_final_metrics_persisted",
        "historical_final_metrics_used_for_fit_or_program_selection",
        "historical_final_used_for_fit_or_program_selection",
        "historical_final_scenes_returned",
        "fresh_selection_conditioned_on_historical_final",
        "fresh_selection_private_overlap_fallback",
        "fresh_selection_historical_scene_fingerprint_overlap",
        "fresh_selection_historical_seed_overlap",
        "fresh_selection_historical_public_observation_overlap",
        "retry_allowed", "formal_ready", "receipt_sha256",
    }),
    "holdout_completed.json": frozenset({
        "version", "status", "campaign_key", "candidate_identity_sha256",
        "attempt_started_sha256", "candidate_authenticated_sha256",
        "historical_exclusion_completed_sha256", "holdout_file_sha256",
        "report_file_sha256", "v3_exclusion_file_sha256",
    }),
    "audit_started.json": frozenset({
        "version", "status", "campaign_key", "candidate_identity_sha256",
        "attempt_started_sha256", "holdout_completed_sha256",
        "candidate_authenticated_sha256", "candidate_artifacts_sha256",
    }),
    "audit_completed.json": frozenset({
        "version", "status", "campaign_key", "candidate_identity_sha256",
        "attempt_started_sha256", "holdout_completed_sha256",
        "candidate_authenticated_sha256", "candidate_artifacts_sha256",
        "report_file_sha256", "evidence_file_sha256",
    }),
}


class _FinalCompletionPrivatePhaseError(RuntimeError):
    """Fixed public error while authenticating fresh-final output bytes."""


class _FinalCompletionPrivatePhaseInterrupt(BaseException):
    """Fixed public interruption while authenticating fresh-final bytes."""
CANDIDATE_ARTIFACT_KEYS = {
    "inputs.json": "candidate_inputs_json",
    "prior_rows_reauthentication_report.json": (
        "candidate_prior_rows_reauthentication_report_json"),
    "prior_v7_rows.npz": "candidate_prior_v7_rows_npz",
    "source_v7_report.json": "candidate_source_v7_report_json",
    "expansion_rows_reauthentication_report.json": (
        "candidate_expansion_rows_reauthentication_report_json"),
    "source_expansion_collection_report.json": (
        "candidate_expansion_source_collection_report_json"),
    "expansion_rows.npz": "candidate_expansion_rows_npz",
    "development_expansion.json": "candidate_development_expansion_json",
    "development_expansion_report.json": (
        "candidate_development_expansion_report_json"),
    "fit_config.json": "candidate_fit_config_json",
    "rows.npz": "development_rows",
    "pairs.npz": "candidate_pairs_npz",
    "weights_audit.json": "candidate_weights_audit_json",
    "program.json": "program",
    "candidate.json": "candidate_candidate_json",
    "report.json": "rcpd_report",
}

# These identities describe the one frozen diagnostic campaign.  Changing the
# program or any producer source cannot create a new campaign key.  A new
# campaign requires a reviewed version change, not a caller-selected input.
EXPECTED_ACTOR_SHA256 = (
    "4ac2ba7782b5556761edaab22bfad50c831c1d8b41b174245e2d81486287ff6b"
)
EXPECTED_PROTOCOL_SHA256 = (
    "374ae398115672a243bd2917937ff056fdc762499b470bf11ad44f5fdecc13c8"
)
EXPECTED_MANIFEST_SHA256 = (
    "af985e9d6f041668ff1250e19da21a078ab5ccc68ecc7aca8d696f2c56845d4c"
)
EXPECTED_DESIGNATION_SHA256 = (
    "b42323e3bc4543c4f4e1af96be4de4d90489a38459240bfb494dcc2d6120a815"
)
EXPECTED_SELECTED_SCENES_SHA256 = (
    "30accfb01d5e022fc42734622cc38481bde639ba9ed2edfb789ddbdf8a6f4fd8"
)
EXPECTED_EXPANSION_REGISTRY_SHA256 = (
    "a687fd3fd4b145ed432af77f3ce26726d4e328df4875e4d69fc3ed351ad98748"
)
EXPECTED_DEVELOPMENT_SUPPLEMENT_SHA256 = (
    "8931b74940f1c41940d9fbb73a52f940f527b8f9c67dfd6b94a4a4d1afd977fe"
)
EXPECTED_RETIRED_IDENTITY_PROJECTION_SHA256 = (
    holdout_api.EXPECTED_RETIRED_IDENTITY_PROJECTION_SHA256)
EXPECTED_RETIRED_IDENTITY_PROJECTION_REPORT_SHA256 = (
    holdout_api.EXPECTED_RETIRED_IDENTITY_PROJECTION_REPORT_SHA256)


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "rcpd_version": audit_api.RCPD_VERSION,
        "holdout_version": holdout_api.VERSION,
        "audit_version": audit_api.VERSION,
        "reservation": (
            "external permanent O_EXCL anchor and campaign registry before "
            "final/program access"
        ),
        "attempts_per_campaign": 1,
        "attempts_per_identity": 1,
        "campaign_key_inputs": [
            "actor_file_sha256", "manifest_file_sha256",
            "designation_file_sha256", "development_expansion_registry_sha256",
            "retired_identity_projection_file_sha256",
            "retired_identity_projection_report_sha256", "holdout_version",
            "selection_salt_commitment",
        ],
        "campaign_key_excludes": [
            "program", "rcpd_report", "producer_sources", "output",
            "ledger_environment", "anchor_environment",
        ],
        "caller_selectable_ledger_or_anchor": False,
        "retry_after_failure": False,
        "retry_after_interruption": False,
        "retry_after_output_removal": False,
        "completion_written_on_failure_where_possible": True,
        "strict_saved_evidence_reader": True,
        "strict_development_candidate_refit_after_claim": True,
        "preclaim_source_closure_snapshot": True,
        "fixed_input_hash_snapshot_after_irrevocable_claim": True,
        "private_selection_salt_access": "post_claim_holdout_only",
        "independent_physical_replay": True,
        "program_fits": 0,
        "actor_updates": 0,
        "runtime_action_override": False,
        "formal_ready": False,
    }


def producer_sources() -> dict[str, str]:
    sources = local_source_hashes((Path(__file__).resolve(),))
    for name, source_sha256 in designation_api.source_closure().items():
        if name in sources and sources[name] != source_sha256:
            raise ValueError("Final-once designation source closure differs")
        sources[name] = source_sha256
    return dict(sorted(sources.items()))


def _ledger_root() -> Path:
    path = Path(REGISTRY_ROOT).expanduser().absolute()
    if path == ROOT.resolve() or path.is_relative_to(ROOT.resolve()):
        raise ValueError("Final-once ledger must be at its fixed repository-external path")
    return path


def _anchor_path() -> Path:
    path = Path(PERMANENT_ANCHOR).expanduser().absolute()
    if path == ROOT.resolve() or path.is_relative_to(ROOT.resolve()):
        raise ValueError("Final-once anchor must be at its fixed repository-external path")
    return path


def _salt_path(value: str | Path | None) -> Path:
    raw = os.environ.get(SALT_FILE_ENV) if value is None else None
    return Path(value if value is not None else (raw or DEFAULT_SALT_FILE)).expanduser().absolute()


def _campaign_identity() -> dict[str, Any]:
    commitment = holdout_api.HOLDOUT_SALT_COMMITMENT
    if (_HEX.fullmatch(str(commitment)) is None
            or commitment == "0" * 64):
        raise ValueError("Private holdout salt commitment is not provisioned")
    return {
        "version": VERSION,
        "actor_file_sha256": EXPECTED_ACTOR_SHA256,
        "manifest_file_sha256": EXPECTED_MANIFEST_SHA256,
        "designation_file_sha256": EXPECTED_DESIGNATION_SHA256,
        "development_expansion_registry_sha256": (
            EXPECTED_EXPANSION_REGISTRY_SHA256
        ),
        "retired_identity_projection_file_sha256": (
            EXPECTED_RETIRED_IDENTITY_PROJECTION_SHA256),
        "retired_identity_projection_report_sha256": (
            EXPECTED_RETIRED_IDENTITY_PROJECTION_REPORT_SHA256),
        "holdout_version": holdout_api.VERSION,
        "selection_salt_commitment": commitment,
    }


def campaign_key() -> str:
    """Return the program/source/output-independent key for the sole campaign."""
    return digest(_campaign_identity())


def _regular(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or path.stat().st_size > MAX_INPUT_BYTES):
        raise ValueError(label + " must be a canonical regular file")
    return path


def _read_json(value: str | Path, label: str) -> dict[str, Any]:
    path = _regular(value, label)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(label + " JSON is oversized")
    # Hash and parse bytes obtained from one O_NOFOLLOW descriptor.  Callers
    # bind the returned object to the independently fixed artifact registry;
    # a pathname hash followed by ``read_text`` would leave an A->B->A window.
    return holdout_api._read_exact_json(path, label)[0]


def _bytes(value: Any) -> bytes:
    return (canonical(value) + "\n").encode("utf-8")


def _write_exclusive(path: Path, raw: bytes) -> None:
    if (path.exists() or path.is_symlink() or path.parent.is_symlink()
            or path.parent.resolve() != path.parent.absolute()):
        raise ValueError("Final-once record destination is unsafe")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _write_phase_payload(stream: Any, raw: bytes) -> None:
    """Finish the private phase payload before it can acquire its final name."""
    stream.write(raw)
    stream.flush()
    os.fsync(stream.fileno())


def _link_no_replace(source: Path, target: Path) -> None:
    """Atomically publish ``source`` and fail if ``target`` already exists."""
    os.link(source, target, follow_symlinks=False)


def _write_phase_exclusive(
    path: Path, raw: bytes, *, before_publish: Callable[[], None] | None = None,
) -> None:
    """Publish a complete phase record without exposing partial final bytes."""
    if (path.exists() or path.is_symlink() or path.parent.is_symlink()
            or path.parent.resolve() != path.parent.absolute()):
        raise ValueError("Final-once phase destination is unsafe")
    temporary = path.parent / ("." + path.name + ".partial")
    descriptor = None
    temporary_created = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        temporary_created = True
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            _write_phase_payload(stream, raw)
        if before_publish is not None:
            before_publish()
        _link_no_replace(temporary, path)
        temporary.unlink()
        temporary_created = False
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_created:
            temporary.unlink(missing_ok=True)


def _input_paths(
    *, actor_path: str | Path, protocol_path: str | Path,
    program_path: str | Path, rcpd_report_path: str | Path,
    manifest_path: str | Path, designation_path: str | Path,
    selected_scenes_path: str | Path,
    development_registry_paths: Sequence[str | Path],
    development_rows_paths: Sequence[str | Path],
    retired_identity_projection_path: str | Path,
) -> tuple[dict[str, Path], list[Path], list[Path], list[Path]]:
    singles = {
        "actor": _regular(actor_path, "frozen Actor"),
        "protocol": _regular(protocol_path, "training protocol"),
        "program": _regular(program_path, "v8 public-tree program"),
        "rcpd_report": _regular(rcpd_report_path, "v8 RCPD report"),
        "manifest": _regular(manifest_path, "conflict manifest"),
        "designation": _regular(designation_path, "Actor designation"),
        "selected_scenes": _regular(selected_scenes_path, "formal X/Y scenes"),
    }
    developments = [_regular(path, "development registry")
                    for path in development_registry_paths]
    rows = [_regular(path, "development rows") for path in development_rows_paths]
    projection = _regular(
        retired_identity_projection_path, "retired identity projection")
    projection_report = _regular(
        projection.parent / "report.json", "retired identity projection report")
    projection_paths = [projection, projection_report]
    if len(developments) != 2 or len(rows) != 1:
        raise ValueError(
            "Final-once requires two development registries, the one v8 rows "
            "artifact, and the fixed retired identity-only projection"
        )
    candidate = singles["program"].parent
    if (singles["program"] != candidate / "program.json"
            or singles["rcpd_report"] != candidate / "report.json"
            or rows[0] != candidate / "rows.npz"):
        raise ValueError(
            "RCPD v8 program, report, and rows must share one candidate directory"
        )
    for name, key in CANDIDATE_ARTIFACT_KEYS.items():
        if key in {"program", "rcpd_report", "development_rows"}:
            continue
        singles[key] = _regular(candidate / name, "embedded RCPD v8 " + name)
    all_paths = [*singles.values(), *developments, *rows, *projection_paths]
    if len({str(path) for path in all_paths}) != len(all_paths):
        raise ValueError("Final-once inputs must be distinct regular files")
    return singles, developments, rows, projection_paths


def _direct_input_hash_snapshot(
    singles: Mapping[str, Path], development_paths: Sequence[Path],
    row_paths: Sequence[Path], projection_paths: Sequence[Path],
    implicit_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Hash every non-private direct input without exposing caller paths.

    The raw selection salt is intentionally absent: the fixed salt commitment
    is part of the campaign identity, while the raw bytes may only be opened by
    the holdout implementation after the irreversible claim exists.
    """

    def checked(path: Path, label: str) -> str:
        return file_hash(_regular(path, label))

    return {
        "singles": {
            name: checked(path, "frozen direct input " + name)
            for name, path in sorted(singles.items())
        },
        "development_registries": [
            checked(path, "frozen development registry")
            for path in development_paths
        ],
        "development_rows": [
            checked(path, "frozen development rows") for path in row_paths
        ],
        "retired_identity_projection": {
            "projection_file_sha256": checked(
                projection_paths[0], "frozen retired identity projection"),
            "projection_report_sha256": checked(
                projection_paths[1], "frozen retired identity projection report"),
        },
        "implicit_files": {
            name: checked(path, "frozen implicit input " + name)
            for name, path in sorted(implicit_paths.items())
        },
    }


def _implicit_input_paths(singles: Mapping[str, Path]) -> dict[str, Path]:
    """Resolve strict-reader inputs which are fixed but not caller arguments."""
    validation = _regular(
        singles["manifest"].parent / "validation.json",
        "frozen manifest validation")
    if file_hash(validation) != manifest_binding.EXPECTED_VALIDATION_SHA256:
        raise ValueError("Exact frozen manifest validation bytes required")
    components = designation_binding.resolve_bound_components(
        singles["designation"], expected_sha256=EXPECTED_DESIGNATION_SHA256)
    return {
        "manifest_validation": validation,
        **{
            "designation_" + name: _regular(
                path, "frozen designation component " + name)
            for name, path in sorted(components.items())
        },
    }


def _guard_frozen_execution_inputs(
    *, singles: Mapping[str, Path], development_paths: Sequence[Path],
    row_paths: Sequence[Path], projection_paths: Sequence[Path],
    implicit_paths: Mapping[str, Path],
    expected_sources: Mapping[str, str],
    expected_inputs: Mapping[str, Any], phase: str,
) -> None:
    """Fail closed if source or any direct input drifts between phases."""
    if producer_sources() != dict(expected_sources):
        raise RuntimeError(
            "Final-once source closure changed during " + phase)
    current_inputs = _direct_input_hash_snapshot(
        singles, development_paths, row_paths, projection_paths, implicit_paths)
    # Check the source closure again because hashing all inputs is not atomic.
    if (current_inputs != dict(expected_inputs)
            or producer_sources() != dict(expected_sources)):
        raise RuntimeError(
            "Final-once direct input or source closure changed during " + phase)


def _preclaim_identity(
    singles: Mapping[str, Path], development_paths: Sequence[Path],
    projection_paths: Sequence[Path], implicit_paths: Mapping[str, Path],
) -> dict[str, Any]:
    """Authenticate only fixed, program-blind campaign inputs before O_EXCL."""
    expected = {
        "actor": EXPECTED_ACTOR_SHA256,
        "protocol": EXPECTED_PROTOCOL_SHA256,
        "designation": EXPECTED_DESIGNATION_SHA256,
        "selected_scenes": EXPECTED_SELECTED_SCENES_SHA256,
    }
    for name, expected_sha256 in expected.items():
        if holdout_api._read_regular_bytes(
                singles[name], "fixed final campaign input " + name,
                maximum=holdout_api.MAX_NPZ_BYTES)[1] != expected_sha256:
            raise ValueError("Fixed final campaign input differs: " + name)
    # Before the private historical-exclusion marker, authenticate only the
    # manifest's exact public file identity.  Full af985 bytes are parsed only
    # by the claim-bound holdout phase.
    if file_hash(singles["manifest"]) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Fixed final campaign input differs: manifest")
    expected_implicit = {
        "manifest_validation": manifest_binding.EXPECTED_VALIDATION_SHA256,
        "designation_actor": designation_api.EXPECTED_ACTOR_SHA256,
        "designation_protocol": designation_api.EXPECTED_PROTOCOL_FILE_SHA256,
        "designation_training_ledger": designation_api.EXPECTED_LEDGER_SHA256,
        "designation_dual_evaluation": (
            designation_api.EXPECTED_DUAL_EVALUATION_SHA256),
        "designation_failure_closeout": designation_api.EXPECTED_CLOSEOUT_SHA256,
    }
    if set(implicit_paths) != set(expected_implicit):
        raise ValueError("Fixed implicit final campaign input set differs")
    for name, expected_sha256 in expected_implicit.items():
        if holdout_api._read_regular_bytes(
                implicit_paths[name], "fixed implicit final input " + name,
                maximum=holdout_api.MAX_NPZ_BYTES)[1] != expected_sha256:
            raise ValueError("Fixed implicit final campaign input differs: " + name)
    if len(projection_paths) != 2:
        raise ValueError("Exact retired identity projection pair required")
    projection_path, projection_report_path = projection_paths
    if (projection_report_path != projection_path.parent / "report.json"
            or file_hash(projection_path)
                != EXPECTED_RETIRED_IDENTITY_PROJECTION_SHA256
            or file_hash(projection_report_path)
                != EXPECTED_RETIRED_IDENTITY_PROJECTION_REPORT_SHA256):
        raise ValueError("Fixed retired identity projection differs")
    _, projection_binding = holdout_api._retired_identity_projection(
        projection_path)
    (_, development_bindings,
     development_values) = holdout_api._development_registry_evidence(
         development_paths)
    supplement_binding = development_bindings.get(
        holdout_api.DEVELOPMENT_SUPPLEMENT_VERSION)
    expansion_binding = development_bindings.get(
        holdout_api.DEVELOPMENT_EXPANSION_VERSION)
    if (not isinstance(supplement_binding, Mapping)
            or supplement_binding.get("file_sha256")
                != EXPECTED_DEVELOPMENT_SUPPLEMENT_SHA256):
        raise ValueError("Fixed development supplement differs")
    if (not isinstance(expansion_binding, Mapping)
            or expansion_binding.get("file_sha256")
                != EXPECTED_EXPANSION_REGISTRY_SHA256):
        raise ValueError("Fixed development expansion registry differs")
    expansion = development_values.get(
        holdout_api.DEVELOPMENT_EXPANSION_VERSION)
    expansion_bindings = (expansion.get("bindings")
                          if isinstance(expansion, Mapping) else None)
    if (not isinstance(expansion, Mapping)
            or not isinstance(expansion_bindings, Mapping)
            or expansion_bindings.get("previous_development_file_sha256")
                != EXPECTED_DEVELOPMENT_SUPPLEMENT_SHA256):
        raise ValueError("Fixed development expansion registry differs")
    holdout_api._validate_retired_expansion_binding(
        projection_binding, expansion)
    return _campaign_identity()


def _candidate_artifact_hashes(
    singles: Mapping[str, Path], row_paths: Sequence[Path],
) -> dict[str, str]:
    if len(row_paths) != 1:
        raise ValueError("Exactly one RCPD v8 rows artifact is required")
    paths = {
        name: (row_paths[0] if key == "development_rows" else singles[key])
        for name, key in CANDIDATE_ARTIFACT_KEYS.items()
    }
    return {name: file_hash(path) for name, path in sorted(paths.items())}


def _validate_candidate_artifact_binding(
    marker: Mapping[str, Any], *, singles: Mapping[str, Path],
    row_paths: Sequence[Path],
) -> dict[str, str]:
    expected = _candidate_artifact_hashes(singles, row_paths)
    frozen = marker.get("candidate_artifacts")
    if (not isinstance(frozen, Mapping) or dict(frozen) != expected
            or set(frozen) != set(CANDIDATE_ARTIFACT_KEYS)
            or marker.get("candidate_artifacts_sha256") != digest(expected)
            or marker.get("program_file_sha256") != expected["program.json"]
            or marker.get("rcpd_report_file_sha256") != expected["report.json"]
            or marker.get("rows_file_sha256") != expected["rows.npz"]
            or marker.get("prior_rows_reauthentication_report_file_sha256")
                != expected["prior_rows_reauthentication_report.json"]
            or marker.get("prior_v7_source_report_file_sha256")
                != expected["source_v7_report.json"]
            or marker.get("prior_v7_rows_file_sha256")
                != expected["prior_v7_rows.npz"]
            or marker.get("expansion_rows_reauthentication_report_file_sha256")
                != expected["expansion_rows_reauthentication_report.json"]
            or marker.get("expansion_source_collection_report_file_sha256")
                != expected["source_expansion_collection_report.json"]
            or marker.get("expansion_rows_file_sha256")
                != expected["expansion_rows.npz"]
            or marker.get("development_expansion_registry_file_sha256")
                != expected["development_expansion.json"]
            or marker.get("development_expansion_report_file_sha256")
                != expected["development_expansion_report.json"]
            or marker.get("fit_config_file_sha256")
                != expected["fit_config.json"]):
        raise ValueError("Claim-authenticated RCPD candidate artifacts changed")
    return expected


def _versioned_json_paths(
    paths: Sequence[Path], *, label: str, expected_versions: set[str],
) -> dict[str, tuple[Path, dict[str, Any]]]:
    result: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in paths:
        value = _read_json(path, label)
        version = value.get("version")
        if version not in expected_versions or version in result:
            raise ValueError(label + " version registry differs")
        result[str(version)] = (path, value)
    if set(result) != expected_versions:
        raise ValueError(label + " version registry differs")
    return result


def _authenticate_candidate_after_claim(
    *, singles: Mapping[str, Path], development_paths: Sequence[Path],
    row_paths: Sequence[Path], projection_paths: Sequence[Path],
) -> tuple[
    dict[str, Any], dict[str, str],
    dict[str, tuple[Path, dict[str, Any]]], dict[str, str],
]:
    """Strictly refit/authenticate the same saved RCPD candidate post-claim."""
    candidate_artifacts = _candidate_artifact_hashes(singles, row_paths)
    developments = _versioned_json_paths(
        development_paths, label="development registry",
        expected_versions={
            holdout_api.DEVELOPMENT_SUPPLEMENT_VERSION,
            holdout_api.DEVELOPMENT_EXPANSION_VERSION,
        },
    )
    if len(projection_paths) != 2:
        raise ValueError("Exact retired identity projection pair required")
    projection_path, projection_report_path = projection_paths
    if projection_report_path != projection_path.parent / "report.json":
        raise ValueError("Retired identity projection report path differs")
    _, projection_binding = holdout_api._retired_identity_projection(
        projection_path)
    expansion_path = developments[holdout_api.DEVELOPMENT_EXPANSION_VERSION][0]
    supplement_path = developments[holdout_api.DEVELOPMENT_SUPPLEMENT_VERSION][0]
    if file_hash(expansion_path) != EXPECTED_EXPANSION_REGISTRY_SHA256:
        raise ValueError("Exact fixed development expansion registry required")
    if (file_hash(projection_path)
            != EXPECTED_RETIRED_IDENTITY_PROJECTION_SHA256
            or file_hash(projection_report_path)
                != EXPECTED_RETIRED_IDENTITY_PROJECTION_REPORT_SHA256):
        raise ValueError("Exact retired identity projection bytes required")
    holdout_api._validate_retired_expansion_binding(
        projection_binding,
        developments[holdout_api.DEVELOPMENT_EXPANSION_VERSION][1],
    )
    report_envelope = _read_json(singles["rcpd_report"], "v8 RCPD report")
    bindings = report_envelope.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("V8 RCPD report bindings differ")
    required_binding_names = (
        "prior_rows_reauthentication_receipt_file_sha256",
        "prior_v7_source_report_file_sha256", "prior_v7_rows_file_sha256",
        "expansion_rows_reauthentication_receipt_file_sha256",
        "expansion_rows_file_sha256", "fit_config_file_sha256",
        "expansion_registry_file_sha256",
        "expansion_registry_report_file_sha256",
        "previous_development_file_sha256",
    )
    if any(_HEX.fullmatch(str(bindings.get(name))) is None
           for name in required_binding_names):
        raise ValueError("V8 RCPD report artifact binding differs")
    candidate_dir = singles["program"].parent
    authenticated = rcpd_api.read_saved_report(
        candidate_dir,
        expected_report_sha256=file_hash(singles["rcpd_report"]),
        actor_path=singles["actor"], protocol_path=singles["protocol"],
        manifest_path=singles["manifest"], designation_path=singles["designation"],
        expansion_registry_path=expansion_path,
        expected_expansion_registry_sha256=EXPECTED_EXPANSION_REGISTRY_SHA256,
        expansion_report_path=singles[
            "candidate_development_expansion_report_json"],
        expected_expansion_report_sha256=str(
            bindings["expansion_registry_report_file_sha256"]),
        expected_prior_rows_report_sha256=str(
            bindings["prior_rows_reauthentication_receipt_file_sha256"]),
        previous_development_path=supplement_path,
        expected_expansion_rows_report_sha256=str(
            bindings["expansion_rows_reauthentication_receipt_file_sha256"]),
        expected_config_sha256=str(bindings["fit_config_file_sha256"]),
        require_passed=True, refit=True,
    )
    designation = _read_json(singles["designation"], "Actor designation")
    evidence = authenticated.get("evidence_artifacts")
    if (authenticated != report_envelope
            or authenticated.get("version") != audit_api.RCPD_VERSION
            or authenticated.get("status") != "passed_development_gates"
            or authenticated.get("explanation_eligible") is not True
            or authenticated.get("formal_ready") is not False
            or not isinstance(evidence, Mapping)
            or dict(evidence) != {
                name: candidate_artifacts[name]
                for name in candidate_artifacts if name != "report.json"
            }
            or evidence.get("program.json") != file_hash(singles["program"])
            or evidence.get("rows.npz") != file_hash(row_paths[0])
            or authenticated.get("program_file_sha256")
                != file_hash(singles["program"])
            or designation.get("bindings", {}).get("actor_sha256")
                != EXPECTED_ACTOR_SHA256
            or designation.get("behavior_performance_gate_waived") is not True
            or designation.get("runtime_action_override") is not False
            or designation.get("formal_ready") is not False
            or designation.get("formal_sample_eligible") is not False):
        raise ValueError("Exact strictly refit passing RCPD v8 candidate required")
    sources = producer_sources()
    if _candidate_artifact_hashes(singles, row_paths) != candidate_artifacts:
        raise RuntimeError(
            "RCPD candidate artifacts changed during strict authentication")
    return deepcopy(authenticated), sources, developments, candidate_artifacts


def _ensure_private_parent(path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (not path.parent.is_dir() or path.parent.is_symlink()
            or path.parent.resolve() != path.parent):
        raise ValueError(label + " parent is unsafe")


def _claim(identity: Mapping[str, Any], *, output: Path) -> tuple[str, Path, dict[str, Any]]:
    expected_identity = _campaign_identity()
    if dict(identity) != expected_identity:
        raise ValueError("Final-once campaign identity differs")
    key = campaign_key()
    identity_sha256 = digest(expected_identity)
    registry_root = _ledger_root()
    anchor = _anchor_path()
    if (anchor == registry_root or anchor.is_relative_to(registry_root)
            or registry_root.is_relative_to(anchor)):
        raise ValueError("Permanent anchor and ledger must be separate paths")
    _ensure_private_parent(anchor, "Permanent final-once anchor")
    anchor_value = {
        "version": VERSION,
        "campaign_key": key,
        "candidate_identity_sha256": identity_sha256,
        "identity": deepcopy(expected_identity),
        "status": "claimed_irrevocable_no_retry",
        "output_requested": str(output),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "automatic_retry": False,
        "output_removal_refunds_attempt": False,
        "formal_ready": False,
    }
    if anchor.exists() or anchor.is_symlink():
        raise ValueError("final_audit_already_reserved_no_retry: " + key)
    try:
        _write_exclusive(anchor, _bytes(anchor_value))
    except FileExistsError as error:
        raise ValueError("final_audit_already_reserved_no_retry: " + key) from error

    # The repository-external permanent anchor is the first irreversible
    # operation.  Everything below may disappear in a crash without refunding
    # the campaign because the anchor remains occupied.
    _ensure_private_parent(registry_root, "Final-once registry")
    try:
        registry_root.mkdir(mode=0o700)
    except FileExistsError:
        if (not registry_root.is_dir() or registry_root.is_symlink()
                or registry_root.resolve() != registry_root):
            raise ValueError("Final-once registry root is unsafe")
    registry = registry_root / key
    try:
        registry.mkdir(mode=0o700)
    except FileExistsError as error:
        raise ValueError("final_audit_already_reserved_no_retry: " + key) from error
    root_fd = os.open(registry_root, os.O_RDONLY)
    try:
        os.fsync(root_fd)
    finally:
        os.close(root_fd)
    started = {
        "version": VERSION,
        "key": key,
        "campaign_key": key,
        "candidate_identity_sha256": identity_sha256,
        "status": "started_irrevocable_no_retry",
        "identity": deepcopy(expected_identity),
        "permanent_anchor_path": str(anchor),
        "permanent_anchor_sha256": file_hash(anchor),
        "output_requested": str(output),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "automatic_retry": False,
        "output_removal_refunds_attempt": False,
        "program_evaluation_started": False,
        "formal_ready": False,
    }
    # This O_EXCL receipt precedes salt access, final selection, strict RCPD
    # refitting, and every explanation-program invocation.
    _write_exclusive(registry / "attempt_started.json", _bytes(started))
    return key, registry, started


def _safe_output(output: str | Path, protected: Sequence[Path]) -> Path:
    path = Path(output).expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise ValueError("Final-once output must be a wholly new directory")
    for item in [_ledger_root(), _anchor_path(), *protected]:
        item = item.absolute()
        if path == item or path.is_relative_to(item) or item.is_relative_to(path):
            raise ValueError("Final-once output overlaps a protected input or ledger")
    ancestor = path.parent
    while not ancestor.exists() and not ancestor.is_symlink():
        ancestor = ancestor.parent
    if ancestor.is_symlink() or not ancestor.is_dir():
        raise ValueError("Final-once output ancestor is unsafe")
    return path


def _output_artifact_paths(
    output: Path, *, include_completion: bool = False,
) -> dict[str, Path]:
    """Return one canonical regular-file registry for a completed output."""
    if not output.exists():
        return {}
    if (not output.is_dir() or output.is_symlink()
            or output.resolve() != output.absolute()):
        raise ValueError("Final-once output directory is unsafe")
    result: dict[str, Path] = {}
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            raise ValueError("Final-once output contains a symbolic link")
        if path.is_dir():
            if path.resolve() != path.absolute():
                raise ValueError("Final-once output directory escaped its root")
            continue
        if not path.is_file() or path.resolve() != path.absolute():
            raise ValueError("Final-once output contains an unsafe artifact")
        name = path.relative_to(output).as_posix()
        if not include_completion and name == "attempt_completed.json":
            continue
        result[name] = path
    return result


def _artifact_hashes(output: Path) -> dict[str, str]:
    return {
        name: file_hash(path)
        for name, path in _output_artifact_paths(output).items()
    }


def _audit_output_attestation(report: Mapping[str, Any]) -> dict[str, str]:
    """Derive exact published audit bytes only from the returned report."""
    artifacts = report.get("evidence_artifacts")
    if (not isinstance(artifacts, Mapping)
            or set(artifacts) != {"inputs.json", "evidence.npz"}
            or any(_HEX.fullmatch(str(value)) is None
                   for value in artifacts.values())):
        raise ValueError("Explanation audit returned no exact artifact attestation")
    return {
        "inputs.json": str(artifacts["inputs.json"]),
        "evidence.npz": str(artifacts["evidence.npz"]),
        "report.json": sha256(_bytes(dict(report))).hexdigest(),
    }


def _snapshot_output_artifacts(
    output: Path, expected: Mapping[str, str], *, prefix: str,
    include_completion: bool = False,
) -> input_snapshot_api.ImmutableInputSnapshot:
    expected_map = {str(name): str(value) for name, value in expected.items()}
    paths = _output_artifact_paths(
        output, include_completion=include_completion)
    if (set(paths) != set(expected_map)
            or any(_HEX.fullmatch(value) is None
                   for value in expected_map.values())):
        raise ValueError("Final-once output artifact registry differs")
    aliases = {f"artifact_{index:03d}": name
               for index, name in enumerate(sorted(paths))}
    return input_snapshot_api.ImmutableInputSnapshot(
        {alias: paths[name] for alias, name in aliases.items()},
        expected_sha256={alias: expected_map[name]
                         for alias, name in aliases.items()},
        relative_names={alias: name for alias, name in aliases.items()},
        maximum_bytes={
            alias: MAX_INPUT_BYTES for alias, name in aliases.items()
            if Path(name).suffix == ".npz"
        },
        prefix=prefix,
    )


def _phase_receipt_hashes(registry: Path) -> dict[str, str]:
    return {
        name: file_hash(registry / name)
        for name in PHASE_RECEIPT_NAMES
        if (registry / name).is_file() and not (registry / name).is_symlink()
    }


def _phase_receipt_values(
    registry: Path,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Read every present phase marker from one immutable byte snapshot."""
    paths = {
        name: _regular(registry / name, "final-once phase " + name)
        for name in PHASE_RECEIPT_NAMES
        if (registry / name).is_file() and not (registry / name).is_symlink()
    }
    expected = {name: file_hash(path) for name, path in paths.items()}
    if not paths:
        return {}, {}
    with input_snapshot_api.ImmutableInputSnapshot(
            paths, expected_sha256=expected,
            relative_names={name: "registry/" + name for name in paths},
            prefix="warehouse-r41-phase-receipts-") as frozen:
        values = {
            name: _read_json(frozen.paths[name], "final-once phase " + name)
            for name in paths
        }
        frozen.verify()
    return expected, values


def _candidate_artifacts_from_marker(value: Mapping[str, Any]) -> dict[str, str]:
    expected_fields = {
        "version", "status", "campaign_key", "candidate_identity_sha256",
        "attempt_started_sha256", "candidate_artifacts",
        "candidate_artifacts_sha256", "rcpd_report_file_sha256",
        "program_file_sha256", "rows_file_sha256",
        "prior_rows_reauthentication_report_file_sha256",
        "prior_v7_source_report_file_sha256", "prior_v7_rows_file_sha256",
        "expansion_rows_reauthentication_report_file_sha256",
        "expansion_source_collection_report_file_sha256",
        "expansion_rows_file_sha256",
        "development_expansion_registry_file_sha256",
        "development_expansion_report_file_sha256", "fit_config_file_sha256",
        "actor_file_sha256", "protocol_file_sha256", "manifest_file_sha256",
        "designation_file_sha256", "selected_scenes_file_sha256",
        "development_registries", "require_passed", "refit", "formal_ready",
    }
    artifacts = value.get("candidate_artifacts")
    development = value.get("development_registries")
    if (set(value) != expected_fields
            or value.get("version") != VERSION + ".candidate-authentication.v1"
            or value.get("status") != "passed_strict_reader_and_refit"
            or not isinstance(artifacts, Mapping)
            or set(artifacts) != set(CANDIDATE_ARTIFACT_KEYS)
            or any(_HEX.fullmatch(str(item)) is None
                   for item in artifacts.values())
            or value.get("candidate_artifacts_sha256") != digest(dict(artifacts))
            or value.get("program_file_sha256") != artifacts.get("program.json")
            or value.get("rcpd_report_file_sha256") != artifacts.get("report.json")
            or value.get("rows_file_sha256") != artifacts.get("rows.npz")
            or value.get("prior_rows_reauthentication_report_file_sha256")
                != artifacts.get("prior_rows_reauthentication_report.json")
            or value.get("prior_v7_source_report_file_sha256")
                != artifacts.get("source_v7_report.json")
            or value.get("prior_v7_rows_file_sha256")
                != artifacts.get("prior_v7_rows.npz")
            or value.get("expansion_rows_reauthentication_report_file_sha256")
                != artifacts.get("expansion_rows_reauthentication_report.json")
            or value.get("expansion_source_collection_report_file_sha256")
                != artifacts.get("source_expansion_collection_report.json")
            or value.get("expansion_rows_file_sha256")
                != artifacts.get("expansion_rows.npz")
            or value.get("development_expansion_registry_file_sha256")
                != artifacts.get("development_expansion.json")
            or value.get("development_expansion_report_file_sha256")
                != artifacts.get("development_expansion_report.json")
            or value.get("fit_config_file_sha256")
                != artifacts.get("fit_config.json")
            or value.get("require_passed") is not True
            or value.get("refit") is not True
            or value.get("formal_ready") is not False):
        raise ValueError("Strict candidate authentication marker differs")
    if (value.get("actor_file_sha256") != EXPECTED_ACTOR_SHA256
            or value.get("protocol_file_sha256") != EXPECTED_PROTOCOL_SHA256
            or value.get("manifest_file_sha256") != EXPECTED_MANIFEST_SHA256
            or value.get("designation_file_sha256") != EXPECTED_DESIGNATION_SHA256
            or value.get("selected_scenes_file_sha256")
                != EXPECTED_SELECTED_SCENES_SHA256
            or not isinstance(development, Mapping)
            or set(development) != {
                holdout_api.DEVELOPMENT_SUPPLEMENT_VERSION,
                holdout_api.DEVELOPMENT_EXPANSION_VERSION,
            }
            or development.get(holdout_api.DEVELOPMENT_EXPANSION_VERSION)
                != EXPECTED_EXPANSION_REGISTRY_SHA256
            or any(_HEX.fullmatch(str(item)) is None
                   for item in development.values())):
        raise ValueError("Strict candidate fixed-input marker differs")
    return {str(name): str(item) for name, item in sorted(artifacts.items())}


def _validate_candidate_marker_before_publish(
    marker: Mapping[str, Any], *, authenticated_report: Mapping[str, Any],
    singles: Mapping[str, Path], development_paths: Sequence[Path],
    row_paths: Sequence[Path], projection_paths: Sequence[Path],
    implicit_paths: Mapping[str, Path],
    identity: Mapping[str, Any], key: str,
    claim_sha256: str,
) -> dict[str, str]:
    """Authenticate the complete marker before its irrevocable publication."""
    artifacts = _candidate_artifacts_from_marker(marker)
    actual_artifacts = _validate_candidate_artifact_binding(
        marker, singles=singles, row_paths=row_paths)
    bindings = authenticated_report.get("bindings")
    developments = marker.get("development_registries")
    supplement_version = holdout_api.DEVELOPMENT_SUPPLEMENT_VERSION
    expansion_version = holdout_api.DEVELOPMENT_EXPANSION_VERSION
    if (not isinstance(bindings, Mapping)
            or not isinstance(developments, Mapping)
            or artifacts != actual_artifacts
            or marker.get("campaign_key") != key
            or key != campaign_key()
            or marker.get("candidate_identity_sha256") != digest(dict(identity))
            or marker.get("attempt_started_sha256") != claim_sha256
            or developments.get(supplement_version)
                != bindings.get("previous_development_file_sha256")
            or developments.get(expansion_version)
                != bindings.get("expansion_registry_file_sha256")
            or _preclaim_identity(
                singles, development_paths, projection_paths,
                implicit_paths) != dict(identity)):
        raise RuntimeError(
            "Complete candidate authentication marker/fixed input changed "
            "before publication")
    return artifacts


def _validate_phase_receipts(
    registry: Path, *, key: str, candidate_identity_sha256: str,
    attempt_started_sha256: str, require_complete: bool,
    terminal_status: str | None = None,
    ignored_receipts: Mapping[str, str] | None = None,
    expected_audit_artifacts: Mapping[str, str] | None = None,
) -> dict[str, str]:
    all_receipts, all_values = _phase_receipt_values(registry)
    ignored = {} if ignored_receipts is None else dict(ignored_receipts)
    if (not set(ignored).issubset({"candidate_authenticated.json"})
            or any(all_receipts.get(name) != value
                   or _HEX.fullmatch(str(value)) is None
                   for name, value in ignored.items())
            or (require_complete and ignored)):
        raise ValueError("Final-once uncommitted phase residue differs")
    receipts = {
        name: value for name, value in all_receipts.items()
        if name not in ignored
    }
    if require_complete and set(receipts) != set(PHASE_RECEIPT_NAMES):
        raise ValueError("Passing final-once chain lacks a phase receipt")
    if not set(receipts).issubset(PHASE_RECEIPT_NAMES):
        raise ValueError("Final-once phase receipt registry differs")
    expected_prefix = set(PHASE_RECEIPT_NAMES[:len(receipts)])
    if set(receipts) != expected_prefix:
        raise ValueError("Final-once phase receipts are not one ordered prefix")
    values = {name: all_values[name] for name in receipts}
    for name, value in values.items():
        if set(value) != _PHASE_FIELDS[name]:
            raise ValueError("Final-once phase receipt schema differs: " + name)
    status_by_name: dict[str, set[str]] = {
        "candidate_authenticated.json": {"passed_strict_reader_and_refit"},
        "holdout_started.json": {"started_no_retry"},
        "historical_exclusion_started.json": {
            "started_irrevocable_no_retry"},
        "historical_exclusion_completed.json": {
            "completed_observation_hash_exclusion"},
        "holdout_completed.json": {"completed_program_blind"},
        "audit_started.json": {"started_no_retry"},
        "audit_completed.json": (
            {"passed"} if terminal_status == "completed_passed"
            else {"passed", "failed"}
        ),
    }
    for name, value in values.items():
        if (value.get("status") not in status_by_name[name]
                or value.get("campaign_key") != key
                or value.get("candidate_identity_sha256")
                    != candidate_identity_sha256
                or value.get("attempt_started_sha256")
                    != attempt_started_sha256):
            raise ValueError("Final-once phase receipt differs: " + name)
    candidate_hash = receipts.get("candidate_authenticated.json")
    candidate_artifacts = None
    if "candidate_authenticated.json" in values:
        candidate_artifacts = _candidate_artifacts_from_marker(
            values["candidate_authenticated.json"])
    if any(name in values for name in PHASE_RECEIPT_NAMES[1:]) and candidate_hash is None:
        raise ValueError("Final-once downstream phase lacks candidate authentication")
    for name in PHASE_RECEIPT_NAMES[1:]:
        if (name in values and values[name].get("candidate_authenticated_sha256")
                != candidate_hash):
            raise ValueError("Final-once candidate phase lineage differs: " + name)
    holdout_started_hash = receipts.get("holdout_started.json")
    for name in (
        "historical_exclusion_started.json",
        "historical_exclusion_completed.json",
    ):
        if (name in values and values[name].get("holdout_started_sha256")
                != holdout_started_hash):
            raise ValueError("Historical exclusion holdout lineage differs")
    historical_started_hash = receipts.get("historical_exclusion_started.json")
    if "historical_exclusion_started.json" in values:
        historical_started = values["historical_exclusion_started.json"]
        started_rows = historical_started.get("row_artifact_evidence")
        candidate_rows = (
            values["candidate_authenticated.json"].get("candidate_artifacts", {})
            if "candidate_authenticated.json" in values else {})
        expected_row_hashes = {
            "merged_rows.npz": candidate_rows.get("rows.npz"),
            "prior_rows.npz": candidate_rows.get("prior_v7_rows.npz"),
            "expansion_rows.npz": candidate_rows.get("expansion_rows.npz"),
        }
        if (historical_started.get("manifest_file_sha256")
                != EXPECTED_MANIFEST_SHA256
                or historical_started.get(
                    "historical_final_access_refunds_attempt") is not False
                or historical_started.get("formal_ready") is not False
                or not isinstance(started_rows, Mapping)
                or set(started_rows) != set(expected_row_hashes)
                or historical_started.get("row_artifact_evidence_sha256")
                    != digest(dict(started_rows))):
            raise ValueError("Historical exclusion start differs")
        for name, expected_file_sha256 in expected_row_hashes.items():
            source = started_rows.get(name)
            if (not isinstance(source, Mapping)
                    or source.get("file_sha256") != expected_file_sha256
                    or type(source.get("row_count")) is not int
                    or source["row_count"] <= 0
                    or type(source.get("unique_scene_fingerprint_count")) is not int
                    or source["unique_scene_fingerprint_count"] <= 0
                    or _HEX.fullmatch(str(source.get(
                        "scene_fingerprints_sha256"))) is None
                    or type(source.get("unique_public_observation_count")) is not int
                    or source["unique_public_observation_count"] <= 0
                    or _HEX.fullmatch(str(source.get(
                        "public_observations_sha256"))) is None):
                raise ValueError("Historical exclusion start row artifact differs")
    if ("historical_exclusion_completed.json" in values
            and values["historical_exclusion_completed.json"].get(
                "historical_exclusion_started_sha256")
                != historical_started_hash):
        raise ValueError("Historical exclusion phase lineage differs")
    historical_completed_hash = receipts.get("historical_exclusion_completed.json")
    if ("holdout_completed.json" in values
            and values["holdout_completed.json"].get(
                "historical_exclusion_completed_sha256")
                != historical_completed_hash):
        raise ValueError("Holdout lacks historical exclusion completion")
    if "historical_exclusion_completed.json" in values:
        historical = values["historical_exclusion_completed.json"]
        claimed_receipt = historical.get("receipt_sha256")
        if (type(claimed_receipt) is not str
                or claimed_receipt != digest({
                    field: item for field, item in historical.items()
                    if field != "receipt_sha256"
                })
                or historical.get("historical_final_scenes_returned") is not False
                or historical.get("fresh_selection_conditioned_on_historical_final")
                    is not False
                or historical.get("fresh_selection_private_overlap_fallback")
                    is not False
                or historical.get(
                    "fresh_selection_historical_scene_fingerprint_overlap") != 0
                or historical.get("fresh_selection_historical_seed_overlap") != 0
                or historical.get(
                    "fresh_selection_historical_public_observation_overlap") != 0
                or historical.get("historical_final_actor_outputs_exposed") is not False
                or historical.get("historical_final_actor_outputs_persisted") is not False
                or historical.get("historical_final_labels_used") is not False
                or historical.get(
                    "historical_final_saved_metrics_replayed_for_authentication")
                    is not False
                or historical.get("historical_final_metrics_exposed") is not False
                or historical.get("historical_final_metrics_persisted") is not False
                or historical.get(
                    "historical_final_metrics_used_for_fit_or_program_selection")
                    is not False
                or historical.get("historical_final_used_for_fit_or_program_selection")
                    is not False
                or historical.get("historical_final_replayed_for_exclusion_only")
                    is not True
                or historical.get(
                    "historical_final_actor_executed_for_observation_exclusion_only")
                    is not True
                or historical.get("retry_allowed") is not False
                or historical.get("formal_ready") is not False
                or historical.get("manifest_file_sha256")
                    != EXPECTED_MANIFEST_SHA256
                or historical.get("historical_final_identity_sha256")
                    != holdout_api.manifest_binding.EXPECTED_FINAL_IDENTITY_SHA256
                or historical.get("historical_final_scene_count")
                    != holdout_api.TOTAL_SCENES
                or type(historical.get(
                    "historical_final_public_observation_count")) is not int
                or historical["historical_final_public_observation_count"] <= 0
                or _HEX.fullmatch(str(historical.get(
                    "historical_final_public_observations_sha256"))) is None
                or type(historical.get(
                    "combined_excluded_scene_fingerprint_count")) is not int
                or historical["combined_excluded_scene_fingerprint_count"]
                    <= holdout_api.TOTAL_SCENES
                or _HEX.fullmatch(str(historical.get(
                    "combined_excluded_scene_fingerprints_sha256"))) is None
                or type(historical.get("combined_excluded_seed_count")) is not int
                or historical["combined_excluded_seed_count"]
                    <= holdout_api.TOTAL_SCENES
                or _HEX.fullmatch(str(historical.get(
                    "combined_excluded_seeds_sha256"))) is None
                or type(historical.get(
                    "combined_forbidden_public_observation_count")) is not int
                or historical["combined_forbidden_public_observation_count"]
                    < historical["historical_final_public_observation_count"]
                or _HEX.fullmatch(str(historical.get(
                    "combined_forbidden_public_observations_sha256"))) is None
                or type(historical.get(
                    "combined_replayed_excluded_public_observation_count")) is not int
                or historical[
                    "combined_replayed_excluded_public_observation_count"]
                    < historical["historical_final_public_observation_count"]
                or _HEX.fullmatch(str(historical.get(
                    "combined_replayed_excluded_public_observations_sha256"))) is None
                or historical.get("development_row_scene_overlap") != 0
                or historical.get("preexisting_scene_identity_overlap") != 0
                or historical.get("preexisting_public_observation_overlap") != 0):
            raise ValueError("Historical exclusion completion differs")
        candidate_rows = (
            values["candidate_authenticated.json"].get("candidate_artifacts", {})
            if "candidate_authenticated.json" in values else {})
        expected_row_hashes = {
            "merged_rows.npz": candidate_rows.get("rows.npz"),
            "prior_rows.npz": candidate_rows.get("prior_v7_rows.npz"),
            "expansion_rows.npz": candidate_rows.get("expansion_rows.npz"),
        }
        started_rows = values["historical_exclusion_started.json"].get(
            "row_artifact_evidence")
        completed_rows = historical.get("row_artifact_evidence")
        if (not isinstance(started_rows, Mapping)
                or not isinstance(completed_rows, Mapping)
                or set(started_rows) != set(expected_row_hashes)
                or set(completed_rows) != set(expected_row_hashes)
                or values["historical_exclusion_started.json"].get(
                    "row_artifact_evidence_sha256") != digest(dict(started_rows))
                or historical.get("row_artifact_evidence_sha256")
                    != digest(dict(completed_rows))):
            raise ValueError("Historical exclusion row lineage differs")
        for name, expected_file_sha256 in expected_row_hashes.items():
            source = started_rows.get(name)
            checked = completed_rows.get(name)
            if (not isinstance(source, Mapping) or not isinstance(checked, Mapping)
                    or checked.get("file_sha256") != expected_file_sha256
                    or source.get("file_sha256") != expected_file_sha256
                    or any(checked.get(field) != source.get(field) for field in (
                        "row_count", "unique_scene_fingerprint_count",
                        "scene_fingerprints_sha256",
                        "unique_public_observation_count",
                        "public_observations_sha256",
                    ))
                    or checked.get("historical_scene_fingerprint_overlap") != 0
                    or checked.get("historical_public_observation_overlap") != 0
                    or checked.get("zero_historical_overlap") is not True):
                raise ValueError("Historical exclusion row artifact differs")
    for name in ("audit_started.json", "audit_completed.json"):
        if (name in values and values[name].get("candidate_artifacts_sha256")
                != digest(candidate_artifacts)):
            raise ValueError("Final-once audit candidate binding differs: " + name)
    holdout_completed_hash = receipts.get("holdout_completed.json")
    for name in ("audit_started.json", "audit_completed.json"):
        if name in values and values[name].get(
                "holdout_completed_sha256") != holdout_completed_hash:
            raise ValueError("Final-once audit phase lineage differs")
    if "audit_completed.json" in values:
        completed = values["audit_completed.json"]
        recorded = {
            "report.json": completed.get("report_file_sha256"),
            "evidence.npz": completed.get("evidence_file_sha256"),
        }
        if any(_HEX.fullmatch(str(value)) is None
               for value in recorded.values()):
            raise ValueError("Final-once audit artifact marker differs")
        if expected_audit_artifacts is not None:
            expected_audit = dict(expected_audit_artifacts)
            if (recorded.get("report.json") != expected_audit.get("report.json")
                    or recorded.get("evidence.npz")
                        != expected_audit.get("evidence.npz")):
                raise ValueError("Final-once audit artifact bytes changed")
    return receipts


def run_final_once(
    *, actor_path: str | Path, protocol_path: str | Path,
    program_path: str | Path, rcpd_report_path: str | Path,
    manifest_path: str | Path, designation_path: str | Path,
    selected_scenes_path: str | Path,
    development_registry_paths: Sequence[str | Path],
    development_rows_paths: Sequence[str | Path],
    retired_identity_projection_path: str | Path, output: str | Path,
    selection_salt_path: str | Path | None = None,
) -> dict[str, Any]:
    singles, development_paths, row_paths, projection_paths = _input_paths(
        actor_path=actor_path, protocol_path=protocol_path,
        program_path=program_path, rcpd_report_path=rcpd_report_path,
        manifest_path=manifest_path, designation_path=designation_path,
        selected_scenes_path=selected_scenes_path,
        development_registry_paths=development_registry_paths,
        development_rows_paths=development_rows_paths,
        retired_identity_projection_path=retired_identity_projection_path,
    )
    salt_path = _salt_path(selection_salt_path)
    protected = [*singles.values(), *development_paths, *row_paths, *projection_paths,
                 salt_path]
    output_path = _safe_output(output, protected)
    # The full manifest contains the protected historical-final split.  Claim
    # the fixed campaign identity before hashing, copying, or parsing any input
    # bytes so an I/O traceback cannot expose those bytes on a retryable path.
    sources = producer_sources()
    identity = _campaign_identity()
    key, registry, started = _claim(identity, output=output_path)
    claim_path = registry / "attempt_started.json"
    claim_sha256 = file_hash(claim_path)
    candidate_identity_sha256 = digest(identity)
    output_created = False
    failed = False
    failure_interrupted = False
    failure_reason: str | None = None
    private_final_phase_started = True
    implicit_paths: dict[str, Path] = {}
    direct_input_snapshot: dict[str, Any] = {}
    validated_phase_receipts: dict[str, str] | None = None
    audit_artifacts: dict[str, str] | None = None
    completion_artifacts: dict[str, str] | None = None
    holdout_report = audit_report = replay_report = candidate_report = None
    candidate_artifacts: dict[str, str] | None = None
    try:
        # Every direct and transitive input is now authenticated under the
        # irreversible, sanitized phase.  None of these reads can refund the
        # attempt or surface its originating traceback.
        implicit_paths = _implicit_input_paths(singles)
        direct_input_snapshot = _direct_input_hash_snapshot(
            singles, development_paths, row_paths, projection_paths,
            implicit_paths)
        _guard_frozen_execution_inputs(
            singles=singles, development_paths=development_paths,
            row_paths=row_paths, projection_paths=projection_paths,
            implicit_paths=implicit_paths,
            expected_sources=sources, expected_inputs=direct_input_snapshot,
            phase="post-claim snapshot")
        if _preclaim_identity(
                singles, development_paths, projection_paths,
                implicit_paths) != identity:
            raise RuntimeError("Fixed final campaign input changed after claim")
        _guard_frozen_execution_inputs(
            singles=singles, development_paths=development_paths,
            row_paths=row_paths, projection_paths=projection_paths,
            implicit_paths=implicit_paths,
            expected_sources=sources, expected_inputs=direct_input_snapshot,
            phase="post-claim authentication")
        output_path.mkdir(parents=True, mode=0o700)
        output_created = True
        _write_exclusive(output_path / "attempt_started.json", _bytes(started))
        # Execute the full strict saved-candidate reader with an authenticated
        # refit only after the fixed post-claim snapshot has passed.
        authenticated_report, authenticated_sources, developments, pending_artifacts = (
            _authenticate_candidate_after_claim(
            singles=singles, development_paths=development_paths,
            row_paths=row_paths, projection_paths=projection_paths,
        ))
        if authenticated_sources != sources:
            raise RuntimeError(
                "Final-once source closure changed during candidate authentication")
        _guard_frozen_execution_inputs(
            singles=singles, development_paths=development_paths,
            row_paths=row_paths, projection_paths=projection_paths,
            implicit_paths=implicit_paths,
            expected_sources=sources, expected_inputs=direct_input_snapshot,
            phase="candidate authentication")
        candidate_marker = {
            "version": VERSION + ".candidate-authentication.v1",
            "status": "passed_strict_reader_and_refit",
            "campaign_key": key,
            "candidate_identity_sha256": candidate_identity_sha256,
            "attempt_started_sha256": claim_sha256,
            "candidate_artifacts": deepcopy(pending_artifacts),
            "candidate_artifacts_sha256": digest(pending_artifacts),
            "rcpd_report_file_sha256": file_hash(singles["rcpd_report"]),
            "program_file_sha256": file_hash(singles["program"]),
            "rows_file_sha256": file_hash(row_paths[0]),
            "prior_rows_reauthentication_report_file_sha256": file_hash(
                singles["candidate_prior_rows_reauthentication_report_json"]),
            "prior_v7_source_report_file_sha256": file_hash(
                singles["candidate_source_v7_report_json"]),
            "prior_v7_rows_file_sha256": file_hash(
                singles["candidate_prior_v7_rows_npz"]),
            "expansion_rows_reauthentication_report_file_sha256": file_hash(
                singles[
                    "candidate_expansion_rows_reauthentication_report_json"]),
            "expansion_source_collection_report_file_sha256": file_hash(
                singles["candidate_expansion_source_collection_report_json"]),
            "expansion_rows_file_sha256": file_hash(
                singles["candidate_expansion_rows_npz"]),
            "development_expansion_registry_file_sha256": file_hash(
                singles["candidate_development_expansion_json"]),
            "development_expansion_report_file_sha256": file_hash(
                singles["candidate_development_expansion_report_json"]),
            "fit_config_file_sha256": file_hash(
                singles["candidate_fit_config_json"]),
            "actor_file_sha256": file_hash(singles["actor"]),
            "protocol_file_sha256": file_hash(singles["protocol"]),
            "manifest_file_sha256": file_hash(singles["manifest"]),
            "designation_file_sha256": file_hash(singles["designation"]),
            "selected_scenes_file_sha256": file_hash(singles["selected_scenes"]),
            "development_registries": {
                version: file_hash(path)
                for version, (path, _) in sorted(developments.items())
            },
            "require_passed": True,
            "refit": True,
            "formal_ready": False,
        }
        if _validate_candidate_marker_before_publish(
                candidate_marker, authenticated_report=authenticated_report,
                singles=singles, development_paths=development_paths,
                row_paths=row_paths, projection_paths=projection_paths,
                implicit_paths=implicit_paths,
                identity=identity, key=key,
                claim_sha256=claim_sha256) != pending_artifacts:
            raise RuntimeError(
                "Strictly authenticated candidate artifact lineage changed")
        _guard_frozen_execution_inputs(
            singles=singles, development_paths=development_paths,
            row_paths=row_paths, projection_paths=projection_paths,
            implicit_paths=implicit_paths,
            expected_sources=sources, expected_inputs=direct_input_snapshot,
            phase="candidate marker publication")
        _write_phase_exclusive(
            registry / "candidate_authenticated.json", _bytes(candidate_marker),
            before_publish=lambda: _guard_frozen_execution_inputs(
                singles=singles, development_paths=development_paths,
                row_paths=row_paths, projection_paths=projection_paths,
                implicit_paths=implicit_paths, expected_sources=sources,
                expected_inputs=direct_input_snapshot,
                phase="staged candidate marker publication"))
        # Only a successfully returned durable marker establishes this
        # lineage.  A write/fsync failure is recorded separately as a burned
        # phase residue and never masquerades as authenticated evidence.
        candidate_report = authenticated_report
        candidate_artifacts = pending_artifacts
        _guard_frozen_execution_inputs(
            singles=singles, development_paths=development_paths,
            row_paths=row_paths, projection_paths=projection_paths,
            implicit_paths=implicit_paths,
            expected_sources=sources, expected_inputs=direct_input_snapshot,
            phase="holdout private phase entry")
        holdout_dir = output_path / "fresh_holdout"
        # From this point forward the holdout may read the committed private
        # salt.  Mark the redaction boundary before entering it so even a
        # failure between salt access and the later historical marker cannot
        # persist or retain an exception payload.
        private_final_phase_started = True
        holdout_report = holdout_api.build(
            actor_path=singles["actor"],
            protocol_path=singles["protocol"],
            manifest_path=singles["manifest"],
            designation_path=singles["designation"],
            selected_scenes_path=singles["selected_scenes"],
            development_registry_paths=development_paths,
            development_rows_paths=row_paths,
            legacy_v3_rows_path=singles["candidate_prior_v7_rows_npz"],
            expansion_rows_path=singles["candidate_expansion_rows_npz"],
            retired_identity_projection_path=projection_paths[0],
            output=holdout_dir,
            claim_receipt_path=claim_path,
            expected_claim_sha256=claim_sha256,
            expected_campaign_key=key,
            expected_candidate_identity_sha256=candidate_identity_sha256,
            selection_salt_path=salt_path,
        )
        holdout_attestation = holdout_report.get("_publication_attestation")
        if (not isinstance(holdout_attestation, Mapping)
                or set(holdout_attestation) != {
                    "artifacts", "holdout_completed_sha256",
                    "historical_exclusion_started_sha256",
                    "historical_exclusion_completed_sha256",
                }
                or not isinstance(holdout_attestation.get("artifacts"), Mapping)
                or set(holdout_attestation["artifacts"]) != {
                    "v3_exclusion.json", "holdout.json", "report.json",
                }
                or any(_HEX.fullmatch(str(value)) is None for value in (
                    *holdout_attestation["artifacts"].values(),
                    holdout_attestation["holdout_completed_sha256"],
                    holdout_attestation["historical_exclusion_started_sha256"],
                    holdout_attestation["historical_exclusion_completed_sha256"],
                ))):
            raise RuntimeError("Fresh-final builder returned no exact attestation")
        persisted_holdout_report = {
            name: value for name, value in holdout_report.items()
            if name != "_publication_attestation"
        }
        _guard_frozen_execution_inputs(
            singles=singles, development_paths=development_paths,
            row_paths=row_paths, projection_paths=projection_paths,
            implicit_paths=implicit_paths,
            expected_sources=sources, expected_inputs=direct_input_snapshot,
            phase="holdout private phase")
        with _snapshot_output_artifacts(
                holdout_dir, holdout_attestation["artifacts"],
                prefix="warehouse-r41-final-holdout-output-") as holdout_snapshot:
            holdout_api.read_saved_holdout(
                holdout_snapshot.root,
                expected_holdout_sha256=holdout_attestation[
                    "artifacts"]["holdout.json"],
                expected_report_sha256=holdout_attestation[
                    "artifacts"]["report.json"],
            )
            frozen_report = _read_json(
                holdout_snapshot.root / "report.json", "fresh-final report")
            if frozen_report != persisted_holdout_report:
                raise RuntimeError("Fresh-final returned report bytes differ")
            holdout_snapshot.verify()
        _guard_frozen_execution_inputs(
            singles=singles, development_paths=development_paths,
            row_paths=row_paths, projection_paths=projection_paths,
            implicit_paths=implicit_paths,
            expected_sources=sources, expected_inputs=direct_input_snapshot,
            phase="holdout authentication")
        holdout_completion = registry / "holdout_completed.json"
        holdout_completion_sha256 = str(
            holdout_attestation["holdout_completed_sha256"])
        phase_hashes, phase_values = _phase_receipt_values(registry)
        completed_holdout = phase_values.get("holdout_completed.json")
        if (phase_hashes.get("holdout_completed.json")
                != holdout_completion_sha256
                or not isinstance(completed_holdout, Mapping)
                or set(completed_holdout)
                    != _PHASE_FIELDS["holdout_completed.json"]
                or completed_holdout.get("holdout_file_sha256")
                    != holdout_attestation["artifacts"]["holdout.json"]
                or completed_holdout.get("report_file_sha256")
                    != holdout_attestation["artifacts"]["report.json"]
                or completed_holdout.get("v3_exclusion_file_sha256")
                    != holdout_attestation["artifacts"]["v3_exclusion.json"]
                or phase_hashes.get("historical_exclusion_started.json")
                    != holdout_attestation[
                        "historical_exclusion_started_sha256"]
                or phase_hashes.get("historical_exclusion_completed.json")
                    != holdout_attestation[
                        "historical_exclusion_completed_sha256"]):
            raise RuntimeError("Fresh-final publication attestation differs")
        expansion_path = developments[holdout_api.DEVELOPMENT_EXPANSION_VERSION][0]
        audit_dir = output_path / "explanation_audit"
        _validate_candidate_artifact_binding(
            _read_json(
                registry / "candidate_authenticated.json",
                "strict candidate authentication phase"),
            singles=singles, row_paths=row_paths)
        _guard_frozen_execution_inputs(
            singles=singles, development_paths=development_paths,
            row_paths=row_paths, projection_paths=projection_paths,
            implicit_paths=implicit_paths,
            expected_sources=sources, expected_inputs=direct_input_snapshot,
            phase="explanation audit entry")
        audit_report = audit_api.audit(
            actor_path=singles["actor"],
            protocol_path=singles["protocol"],
            program_path=singles["program"],
            rcpd_report_path=singles["rcpd_report"],
            manifest_path=singles["manifest"],
            designation_path=singles["designation"],
            development_expansion_path=expansion_path,
            fresh_holdout_path=holdout_dir / "holdout.json",
            fresh_holdout_report_path=holdout_dir / "report.json",
            development_rows_paths=row_paths,
            output=audit_dir,
            claim_receipt_path=claim_path,
            expected_claim_sha256=claim_sha256,
            expected_campaign_key=key,
            expected_candidate_identity_sha256=candidate_identity_sha256,
            expected_holdout_completion_sha256=holdout_completion_sha256,
        )
        _guard_frozen_execution_inputs(
            singles=singles, development_paths=development_paths,
            row_paths=row_paths, projection_paths=projection_paths,
            implicit_paths=implicit_paths,
            expected_sources=sources, expected_inputs=direct_input_snapshot,
            phase="explanation audit")
        # Bind the precise bytes emitted by the trusted audit call before any
        # saved reader or physical replay.  Expected hashes come exclusively
        # from the returned in-memory report, never from the live output path.
        audit_artifacts = _audit_output_attestation(audit_report)
        with _snapshot_output_artifacts(
                audit_dir, audit_artifacts,
                prefix="warehouse-r41-final-audit-output-") as audit_snapshot:
            frozen_audit_dir = audit_snapshot.root
            strict_audit_report = audit_api.read_saved_report(
                frozen_audit_dir,
                expected_report_sha256=audit_artifacts["report.json"],
                program_path=singles["program"],
                development_rows_paths=row_paths,
                expected_bindings=audit_report["bindings"],
                require_passed=True,
            )
            if strict_audit_report.get("status") != "passed":
                raise RuntimeError("Explanation audit strict reader did not pass")
            _guard_frozen_execution_inputs(
                singles=singles, development_paths=development_paths,
                row_paths=row_paths, projection_paths=projection_paths,
                implicit_paths=implicit_paths,
                expected_sources=sources, expected_inputs=direct_input_snapshot,
                phase="explanation audit authentication")
            replay_report = audit_api.replay_saved_audit(
                frozen_audit_dir,
                actor_path=singles["actor"],
                protocol_path=singles["protocol"],
                program_path=singles["program"],
                manifest_path=singles["manifest"],
                fresh_holdout_path=holdout_dir / "holdout.json",
                expected_evidence_sha256=audit_artifacts["evidence.npz"],
                expected_bindings=audit_report["bindings"],
            )
            audit_snapshot.verify()
        _guard_frozen_execution_inputs(
            singles=singles, development_paths=development_paths,
            row_paths=row_paths, projection_paths=projection_paths,
            implicit_paths=implicit_paths,
            expected_sources=sources, expected_inputs=direct_input_snapshot,
            phase="physical replay")
        _validate_candidate_artifact_binding(
            _read_json(
                registry / "candidate_authenticated.json",
                "strict candidate authentication phase"),
            singles=singles, row_paths=row_paths)
        _guard_frozen_execution_inputs(
            singles=singles, development_paths=development_paths,
            row_paths=row_paths, projection_paths=projection_paths,
            implicit_paths=implicit_paths,
            expected_sources=sources, expected_inputs=direct_input_snapshot,
            phase="physical replay publication")
        replay_raw = _bytes(replay_report)
        replay_sha256 = sha256(replay_raw).hexdigest()
        _write_exclusive(output_path / "physical_replay.json", replay_raw)
        validated_phase_receipts = _validate_phase_receipts(
            registry, key=key,
            candidate_identity_sha256=candidate_identity_sha256,
            attempt_started_sha256=claim_sha256, require_complete=True,
            terminal_status="completed_passed",
            expected_audit_artifacts=audit_artifacts)
        completion_artifacts = {
            "attempt_started.json": claim_sha256,
            **{
                "fresh_holdout/" + name: str(value)
                for name, value in holdout_attestation["artifacts"].items()
            },
            **{
                "explanation_audit/" + name: value
                for name, value in audit_artifacts.items()
            },
            "physical_replay.json": replay_sha256,
        }
        if (set(completion_artifacts) != _SUCCESS_OUTPUT_ARTIFACT_NAMES
                or _artifact_hashes(output_path) != completion_artifacts):
            raise RuntimeError("Successful final-once output registry differs")
        _guard_frozen_execution_inputs(
            singles=singles, development_paths=development_paths,
            row_paths=row_paths, projection_paths=projection_paths,
            implicit_paths=implicit_paths,
            expected_sources=sources, expected_inputs=direct_input_snapshot,
            phase="final completion")
    except BaseException as error:
        # Compress the exception while still in the handler, then let Python
        # clear ``error`` before any receipt hashing or publication work can
        # itself fail.  No downstream traceback can retain the primary one.
        failed = True
        failure_interrupted = not isinstance(error, Exception)
        failure_reason = (
            "historical_final_private_phase_interrupted"
            if private_final_phase_started and failure_interrupted
            else "historical_final_private_phase_failed"
            if private_final_phase_started
            else f"{type(error).__name__}: {error}"
        )

    status = "completed_passed" if not failed else (
        "burned_interrupted" if failure_interrupted else "burned_failed")
    phase_receipts = (
        dict(validated_phase_receipts)
        if not failed and validated_phase_receipts is not None
        else _phase_receipt_hashes(registry)
    )
    phase_residues: dict[str, str] = {}
    if (candidate_report is None
            and "candidate_authenticated.json" in phase_receipts):
        phase_residues["candidate_authenticated.json"] = phase_receipts.pop(
            "candidate_authenticated.json")
    completion = {
        "version": VERSION,
        "key": key,
        "campaign_key": key,
        "candidate_identity_sha256": candidate_identity_sha256,
        "status": status,
        "identity": deepcopy(identity),
        "producer_sources": sources,
        "attempt_started_sha256": claim_sha256,
        "permanent_anchor_sha256": started["permanent_anchor_sha256"],
        "phase_receipts": phase_receipts,
        "uncommitted_phase_residues": phase_residues,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "automatic_retry": False,
        "retry_allowed": False,
        "reason": failure_reason,
        "output_created": output_created,
        "output_identity": str(output_path),
        "candidate_authentication_status": (
            None if candidate_report is None else "passed_strict_reader_and_refit"
        ),
        "candidate_authenticated_sha256": (
            phase_receipts.get("candidate_authenticated.json")
            if candidate_report is not None else None
        ),
        "candidate_artifacts": deepcopy(candidate_artifacts),
        "candidate_artifacts_sha256": (
            None if candidate_artifacts is None else digest(candidate_artifacts)
        ),
        "holdout_status": None if holdout_report is None else holdout_report.get("status"),
        "audit_status": None if audit_report is None else audit_report.get("status"),
        "physical_replay_status": None if replay_report is None else replay_report.get("status"),
        "artifacts": (
            dict(completion_artifacts)
            if not failed and completion_artifacts is not None
            # A failed attempt must never reopen fresh-final output after the
            # private exception boundary.  Besides avoiding an untrusted
            # partial registry, this keeps raw holdout bytes out of any
            # secondary file-hash traceback.
            else {}
        ),
        "program_fits": 0,
        "development_authentication_refit": candidate_report is not None,
        "actor_updates": 0,
        "runtime_action_override": False,
        "formal_ready": False,
    }
    completion_raw = _bytes(completion)
    completion_sha256 = sha256(completion_raw).hexdigest()
    completion_error = None
    try:
        def completion_guard() -> None:
            _guard_frozen_execution_inputs(
                singles=singles, development_paths=development_paths,
                row_paths=row_paths, projection_paths=projection_paths,
                implicit_paths=implicit_paths, expected_sources=sources,
                expected_inputs=direct_input_snapshot,
                phase="staged successful completion publication")
            current = _validate_phase_receipts(
                registry, key=key,
                candidate_identity_sha256=candidate_identity_sha256,
                attempt_started_sha256=claim_sha256, require_complete=True,
                terminal_status="completed_passed",
                expected_audit_artifacts=audit_artifacts)
            if current != validated_phase_receipts:
                raise RuntimeError(
                    "Final-once phase receipts changed before completion publication")
            if (completion_artifacts is None
                    or _artifact_hashes(output_path) != completion_artifacts):
                raise RuntimeError(
                    "Final-once output changed before completion publication")

        _write_phase_exclusive(
            registry / "attempt_completed.json", completion_raw,
            before_publish=(completion_guard if not failed else None))
        if output_created:
            _write_phase_exclusive(output_path / "attempt_completed.json", completion_raw)
    except BaseException as error:
        completion_error = error
    if completion_error is not None:
        # A secondary publication error must not retain a primary private
        # failure object in this frame.  Recreate only a fixed/public error
        # after both exception objects have been dropped.
        primary_failure_present = failed
        completion_interrupted = not isinstance(completion_error, Exception)
        completion_message = (
            None if primary_failure_present else str(completion_error))
        completion_error = None
        if completion_interrupted:
            raise holdout_api._HistoricalFinalPrivatePhaseInterrupt(
                "final_once_completion_publication_interrupted") from None
        if primary_failure_present:
            raise RuntimeError(
                "final_once_completion_publication_failed") from None
        raise RuntimeError(str(completion_message)) from None
    result = {
        "status": status,
        "key": key,
        "registry": str(registry),
        "output": str(output_path),
        "completion_sha256": (
            completion_sha256
            if (registry / "attempt_completed.json").is_file() else None
        ),
        "reason": completion["reason"],
        "retry_allowed": False,
        "formal_ready": False,
    }
    if not failed:
        authenticated_completion = read_completion(
            registry,
            expected_completion_sha256=result["completion_sha256"],
            expected_identity=identity)
        if (authenticated_completion.get("status") != "completed_passed"
                or authenticated_completion.get("phase_receipts")
                    != validated_phase_receipts):
            raise RuntimeError("Final-once completed receipt reread differs")
    if failed and failure_interrupted:
        interrupted_private_phase = private_final_phase_started
        if interrupted_private_phase:
            raise holdout_api._HistoricalFinalPrivatePhaseInterrupt(
                "historical_final_private_phase_interrupted") from None
        raise holdout_api._HistoricalFinalPrivatePhaseInterrupt(
            "final_once_interrupted") from None
    return result


def read_completion(registry: str | Path, *, expected_completion_sha256: str,
                    expected_identity: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(registry).expanduser().absolute()
    key = campaign_key()
    if (dict(expected_identity) != _campaign_identity()
            or path != _ledger_root() / key
            or not path.is_dir() or path.is_symlink() or path.resolve() != path
            or _HEX.fullmatch(expected_completion_sha256) is None):
        raise ValueError("Final-once registry/completion anchor differs")
    original_started = _regular(path / "attempt_started.json", "attempt start")
    original_completion = _regular(
        path / "attempt_completed.json", "attempt completion")
    original_anchor = _regular(_anchor_path(), "permanent final-once anchor")
    sources = producer_sources()
    originals: dict[str, Path] = {
        "attempt_started": original_started,
        "attempt_completed": original_completion,
        "permanent_anchor": original_anchor,
    }
    relative_names = {
        "attempt_started": "registry/attempt_started.json",
        "attempt_completed": "registry/attempt_completed.json",
        "permanent_anchor": "anchor/permanent_anchor.json",
    }
    for name in PHASE_RECEIPT_NAMES:
        candidate = path / name
        if candidate.is_file() and not candidate.is_symlink():
            snapshot_name = "phase_" + name.removesuffix(".json")
            originals[snapshot_name] = _regular(
                candidate, "final-once phase " + name)
            relative_names[snapshot_name] = "registry/" + name
    expected = {name: file_hash(value) for name, value in originals.items()}
    expected["attempt_completed"] = expected_completion_sha256
    with input_snapshot_api.ImmutableInputSnapshot(
            originals, expected_sha256=expected,
            relative_names=relative_names,
            prefix="warehouse-r41-final-completion-") as frozen:
        registry_snapshot = frozen.root / "registry"
        started_path = frozen.paths["attempt_started"]
        completion_path = frozen.paths["attempt_completed"]
        anchor_path = frozen.paths["permanent_anchor"]
        started = _read_json(started_path, "attempt start")
        completion = _read_json(completion_path, "attempt completion")
        anchor = _read_json(anchor_path, "permanent final-once anchor")
        if set(completion) != _COMPLETION_FIELDS:
            raise ValueError("Final-once completion schema differs")
        identity_sha256 = digest(expected_identity)
        phase_receipts = completion.get("phase_receipts")
        actual_phase_receipts = _validate_phase_receipts(
            registry_snapshot, key=key,
            candidate_identity_sha256=identity_sha256,
            attempt_started_sha256=file_hash(started_path),
            require_complete=completion.get("status") == "completed_passed",
            terminal_status=completion.get("status"),
            ignored_receipts=completion.get("uncommitted_phase_residues", {}))
        if (not isinstance(phase_receipts, Mapping)
                or dict(phase_receipts) != actual_phase_receipts):
            raise ValueError("Final-once phase receipt registry differs")
        candidate_marker_path = registry_snapshot / "candidate_authenticated.json"
        candidate_is_residue = candidate_marker_path.name in completion.get(
            "uncommitted_phase_residues", {})
        if (candidate_marker_path.is_file()
                and not candidate_marker_path.is_symlink()
                and not candidate_is_residue):
            marker = _read_json(
                candidate_marker_path, "strict candidate authentication phase")
            frozen_candidate = _candidate_artifacts_from_marker(marker)
            if (completion.get("candidate_authenticated_sha256")
                    != file_hash(candidate_marker_path)
                    or completion.get("candidate_artifacts") != frozen_candidate
                    or completion.get("candidate_artifacts_sha256")
                        != digest(frozen_candidate)):
                raise ValueError("Final-once completion candidate binding differs")
        elif (not candidate_is_residue
              and (completion.get("candidate_authenticated_sha256") is not None
              or completion.get("candidate_artifacts") is not None
              or completion.get("candidate_artifacts_sha256") is not None)):
            raise ValueError("Final-once completion has an unauthenticated candidate")
        if candidate_is_residue and (
                completion.get("status") == "completed_passed"
                or completion.get("candidate_authenticated_sha256") is not None
                or completion.get("candidate_artifacts") is not None
                or completion.get("candidate_artifacts_sha256") is not None):
            raise ValueError(
                "Final-once candidate residue was treated as authenticated")
        if (started.get("key") != key or started.get("campaign_key") != key
                or completion.get("key") != key
                or completion.get("campaign_key") != key
                or anchor.get("campaign_key") != key
                or started.get("candidate_identity_sha256") != identity_sha256
                or completion.get("candidate_identity_sha256") != identity_sha256
                or anchor.get("candidate_identity_sha256") != identity_sha256
                or started.get("identity") != dict(expected_identity)
                or completion.get("identity") != dict(expected_identity)
                or anchor.get("identity") != dict(expected_identity)
                or started.get("status") != "started_irrevocable_no_retry"
                or anchor.get("status") != "claimed_irrevocable_no_retry"
                or completion.get("status") not in {
                    "completed_passed", "burned_failed", "burned_interrupted"}
                or completion.get("attempt_started_sha256")
                    != file_hash(started_path)
                or started.get("permanent_anchor_sha256") != file_hash(anchor_path)
                or completion.get("permanent_anchor_sha256")
                    != file_hash(anchor_path)
                or completion.get("automatic_retry") is not False
                or completion.get("retry_allowed") is not False
                or started.get("output_requested") != anchor.get("output_requested")
                or completion.get("output_identity") != started.get("output_requested")
                or (completion.get("status") == "completed_passed"
                    and completion.get("producer_sources") != sources)):
            raise ValueError("Final-once permanent record differs")
        artifacts = completion.get("artifacts")
        if (not isinstance(artifacts, Mapping)
                or any(_HEX.fullmatch(str(value)) is None
                       for value in artifacts.values())):
            raise ValueError("Final-once completion artifact registry differs")
        if completion.get("status") == "completed_passed":
            if (completion.get("reason") is not None
                    or completion.get("output_created") is not True
                    or completion.get("candidate_authentication_status")
                        != "passed_strict_reader_and_refit"
                    or completion.get("holdout_status")
                        != "passed_program_blind_registry"
                    or completion.get("audit_status") != "passed"
                    or completion.get("physical_replay_status") != "passed"
                    or completion.get("program_fits") != 0
                    or completion.get("development_authentication_refit") is not True
                    or completion.get("actor_updates") != 0
                    or completion.get("runtime_action_override") is not False
                    or completion.get("formal_ready") is not False
                    or set(artifacts) != _SUCCESS_OUTPUT_ARTIFACT_NAMES):
                raise ValueError("Passing final-once completion invariants differ")
            output_path = Path(str(completion["output_identity"])).expanduser().absolute()
            if str(output_path) != completion["output_identity"]:
                raise ValueError("Final-once output identity is not canonical")
            complete_output_artifacts = dict(artifacts)
            complete_output_artifacts["attempt_completed.json"] = (
                expected_completion_sha256)
            def verify_private_output() -> None:
                with _snapshot_output_artifacts(
                        output_path, complete_output_artifacts,
                        prefix="warehouse-r41-completion-output-",
                        include_completion=True) as output_snapshot:
                    audit_marker = _read_json(
                        registry_snapshot / "audit_completed.json",
                        "audit completion phase")
                    if (audit_marker.get("report_file_sha256")
                            != artifacts.get("explanation_audit/report.json")
                            or audit_marker.get("evidence_file_sha256")
                                != artifacts.get("explanation_audit/evidence.npz")):
                        raise ValueError(
                            "Final-once audit marker/output binding differs")
                    output_snapshot.verify()

            private_output_failure = False
            private_output_interrupted = False
            try:
                verify_private_output()
            except BaseException as error:
                private_output_failure = True
                private_output_interrupted = not isinstance(error, Exception)
            finally:
                verify_private_output = None
            if private_output_failure:
                if private_output_interrupted:
                    raise _FinalCompletionPrivatePhaseInterrupt(
                        "final_completion_private_phase_interrupted") from None
                raise _FinalCompletionPrivatePhaseError(
                    "final_completion_private_phase_failed") from None
        frozen.verify()
        if producer_sources() != sources:
            raise RuntimeError(
                "Final-once source closure changed during completion read")
        return deepcopy(completion)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--program", required=True)
    parser.add_argument("--rcpd-report", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--designation", required=True)
    parser.add_argument("--selected-scenes", required=True)
    parser.add_argument("--development-registry", action="append", required=True)
    parser.add_argument("--development-rows", action="append", required=True)
    parser.add_argument("--retired-identity-projection", required=True)
    parser.add_argument(
        "--selection-salt",
        help=("path to the private raw salt; defaults to $" + SALT_FILE_ENV),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = run_final_once(
        actor_path=args.actor, protocol_path=args.protocol,
        program_path=args.program, rcpd_report_path=args.rcpd_report,
        manifest_path=args.manifest, designation_path=args.designation,
        selected_scenes_path=args.selected_scenes,
        development_registry_paths=args.development_registry,
        development_rows_paths=args.development_rows,
        retired_identity_projection_path=args.retired_identity_projection,
        output=args.output,
        selection_salt_path=args.selection_salt,
    )
    print(canonical(result))
    return 0 if result["status"] == "completed_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VERSION", "REGISTRY_ROOT", "PERMANENT_ANCHOR", "LEDGER_ENV",
    "PERMANENT_ANCHOR_ENV", "SALT_FILE_ENV", "contract", "producer_sources",
    "campaign_key", "run_final_once", "read_completion", "main",
]
