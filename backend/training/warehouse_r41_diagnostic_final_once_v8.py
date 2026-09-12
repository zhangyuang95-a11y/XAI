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
from typing import Any, Mapping, Sequence

from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training import warehouse_r41_diagnostic_fresh_final_holdout_v4 as holdout_api
from backend.training import warehouse_r41_diagnostic_explanation_audit_v8 as audit_api
from backend.training import warehouse_r41_diagnostic_rcpd_v8 as rcpd_api


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
    "holdout_completed.json", "audit_started.json", "audit_completed.json",
)
CANDIDATE_ARTIFACT_KEYS = {
    "program.json": "program",
    "report.json": "rcpd_report",
    "rows.npz": "development_rows",
    "prior_v7_report.json": "candidate_prior_v7_report_json",
    "prior_v7_rows.npz": "candidate_prior_v7_rows_npz",
    "expansion_rows.npz": "candidate_expansion_rows_npz",
    "fit_config.json": "candidate_fit_config_json",
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
    "d80f2736c6d5e359764f6ae4c09ccfa277f26e98a8910c1f18d86d84cb99851f"
)
EXPECTED_SELECTED_SCENES_SHA256 = (
    "30accfb01d5e022fc42734622cc38481bde639ba9ed2edfb789ddbdf8a6f4fd8"
)
EXPECTED_EXPANSION_REGISTRY_SHA256 = (
    "abbced9b99eb51857ad36c2d7b9351864c4472f5f9de922228d5744ae4d49d0a"
)
EXPECTED_RETIRED_HOLDOUT_SHA256 = dict(
    holdout_api.EXPECTED_RETIRED_HOLDOUT_SHA256)


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
            "retired_holdout_sha256", "holdout_version",
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
        "independent_physical_replay": True,
        "program_fits": 0,
        "actor_updates": 0,
        "runtime_action_override": False,
        "formal_ready": False,
    }


def producer_sources() -> dict[str, str]:
    return dict(sorted(local_source_hashes((Path(__file__).resolve(),)).items()))


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
        "retired_holdout_sha256": dict(
            sorted(EXPECTED_RETIRED_HOLDOUT_SHA256.items())),
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

    def pairs(rows):
        result = {}
        for key, item in rows:
            if key in result:
                raise ValueError("Duplicate JSON field in " + label)
            result[key] = item
        return result

    result = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError("Non-finite JSON value in " + label + ": " + token)
        ),
    )
    if not isinstance(result, dict):
        raise ValueError(label + " must be one JSON object")
    return result


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


def _write_phase_exclusive(path: Path, raw: bytes) -> None:
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
    retired_holdout_paths: Sequence[str | Path],
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
    retired = [_regular(path, "retired fresh-final registry")
               for path in retired_holdout_paths]
    if len(developments) != 2 or len(rows) != 1 or len(retired) != 2:
        raise ValueError(
            "Final-once requires two development registries, the one v8 rows "
            "artifact, and retired v1/v2 only"
        )
    candidate = singles["program"].parent
    if (singles["program"] != candidate / "program.json"
            or singles["rcpd_report"] != candidate / "report.json"
            or rows[0] != candidate / "rows.npz"):
        raise ValueError(
            "RCPD v8 program, report, and rows must share one candidate directory"
        )
    for name in ("prior_v7_report.json", "prior_v7_rows.npz",
                 "expansion_rows.npz", "fit_config.json"):
        singles["candidate_" + name.replace(".", "_")] = _regular(
            candidate / name, "embedded RCPD v8 " + name)
    all_paths = [*singles.values(), *developments, *rows, *retired]
    if len({str(path) for path in all_paths}) != len(all_paths):
        raise ValueError("Final-once inputs must be distinct regular files")
    return singles, developments, rows, retired


def _preclaim_identity(
    singles: Mapping[str, Path], development_paths: Sequence[Path],
) -> dict[str, Any]:
    """Authenticate only fixed, program-blind campaign inputs before O_EXCL."""
    expected = {
        "actor": EXPECTED_ACTOR_SHA256,
        "protocol": EXPECTED_PROTOCOL_SHA256,
        "manifest": EXPECTED_MANIFEST_SHA256,
        "designation": EXPECTED_DESIGNATION_SHA256,
        "selected_scenes": EXPECTED_SELECTED_SCENES_SHA256,
    }
    for name, expected_sha256 in expected.items():
        if file_hash(singles[name]) != expected_sha256:
            raise ValueError("Fixed final campaign input differs: " + name)
    development_hashes = {file_hash(path) for path in development_paths}
    if EXPECTED_EXPANSION_REGISTRY_SHA256 not in development_hashes:
        raise ValueError("Fixed development expansion registry differs")
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
            or marker.get("prior_v7_report_file_sha256")
                != expected["prior_v7_report.json"]
            or marker.get("prior_v7_rows_file_sha256")
                != expected["prior_v7_rows.npz"]
            or marker.get("expansion_rows_file_sha256")
                != expected["expansion_rows.npz"]
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
    row_paths: Sequence[Path], retired_paths: Sequence[Path],
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
    retired = _versioned_json_paths(
        retired_paths, label="retired fresh-final registry",
        expected_versions=set(holdout_api.EXTERNAL_RETIRED_VERSIONS),
    )
    expansion_path = developments[holdout_api.DEVELOPMENT_EXPANSION_VERSION][0]
    supplement_path = developments[holdout_api.DEVELOPMENT_SUPPLEMENT_VERSION][0]
    if file_hash(expansion_path) != EXPECTED_EXPANSION_REGISTRY_SHA256:
        raise ValueError("Exact fixed development expansion registry required")
    actual_retired = {
        version: {
            "file_sha256": file_hash(path),
            "content_sha256": value.get("content_sha256"),
        }
        for version, (path, value) in sorted(retired.items())
    }
    expansion_retired = developments[
        holdout_api.DEVELOPMENT_EXPANSION_VERSION][1].get(
            "bindings", {}).get("retired_holdouts")
    if (any(actual_retired[version]["file_sha256"]
            != EXPECTED_RETIRED_HOLDOUT_SHA256[version]
            for version in holdout_api.EXTERNAL_RETIRED_VERSIONS)
            or actual_retired != expansion_retired):
        raise ValueError(
            "Retired v1/v2 bytes must match the fixed development expansion")
    report_envelope = _read_json(singles["rcpd_report"], "v8 RCPD report")
    bindings = report_envelope.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("V8 RCPD report bindings differ")
    required_binding_names = (
        "prior_v7_report_file_sha256", "expansion_rows_file_sha256",
        "fit_config_file_sha256", "expansion_registry_file_sha256",
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
        expected_prior_v7_report_sha256=str(
            bindings["prior_v7_report_file_sha256"]),
        previous_development_path=supplement_path,
        expected_expansion_rows_sha256=str(bindings["expansion_rows_file_sha256"]),
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


def _artifact_hashes(output: Path) -> dict[str, str]:
    result = {}
    if not output.is_dir() or output.is_symlink():
        return result
    for path in sorted(output.rglob("*")):
        if path.is_file() and not path.is_symlink():
            result[path.relative_to(output).as_posix()] = file_hash(path)
    return result


def _phase_receipt_hashes(registry: Path) -> dict[str, str]:
    return {
        name: file_hash(registry / name)
        for name in PHASE_RECEIPT_NAMES
        if (registry / name).is_file() and not (registry / name).is_symlink()
    }


def _candidate_artifacts_from_marker(value: Mapping[str, Any]) -> dict[str, str]:
    expected_fields = {
        "version", "status", "campaign_key", "candidate_identity_sha256",
        "attempt_started_sha256", "candidate_artifacts",
        "candidate_artifacts_sha256", "rcpd_report_file_sha256",
        "program_file_sha256", "rows_file_sha256",
        "prior_v7_report_file_sha256", "prior_v7_rows_file_sha256",
        "expansion_rows_file_sha256", "fit_config_file_sha256",
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
            or value.get("prior_v7_report_file_sha256")
                != artifacts.get("prior_v7_report.json")
            or value.get("prior_v7_rows_file_sha256")
                != artifacts.get("prior_v7_rows.npz")
            or value.get("expansion_rows_file_sha256")
                != artifacts.get("expansion_rows.npz")
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
    row_paths: Sequence[Path], identity: Mapping[str, Any], key: str,
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
            or _preclaim_identity(singles, development_paths) != dict(identity)):
        raise RuntimeError(
            "Complete candidate authentication marker/fixed input changed "
            "before publication")
    return artifacts


def _validate_phase_receipts(
    registry: Path, *, key: str, candidate_identity_sha256: str,
    attempt_started_sha256: str, require_complete: bool,
    terminal_status: str | None = None,
    ignored_receipts: Mapping[str, str] | None = None,
) -> dict[str, str]:
    all_receipts = _phase_receipt_hashes(registry)
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
    values = {
        name: _read_json(registry / name, "final-once phase " + name)
        for name in receipts
    }
    status_by_name: dict[str, set[str]] = {
        "candidate_authenticated.json": {"passed_strict_reader_and_refit"},
        "holdout_started.json": {"started_no_retry"},
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
    for name in ("audit_started.json", "audit_completed.json"):
        if (name in values and values[name].get("candidate_artifacts_sha256")
                != digest(candidate_artifacts)):
            raise ValueError("Final-once audit candidate binding differs: " + name)
    holdout_completed_hash = receipts.get("holdout_completed.json")
    for name in ("audit_started.json", "audit_completed.json"):
        if name in values and values[name].get(
                "holdout_completed_sha256") != holdout_completed_hash:
            raise ValueError("Final-once audit phase lineage differs")
    return receipts


def run_final_once(
    *, actor_path: str | Path, protocol_path: str | Path,
    program_path: str | Path, rcpd_report_path: str | Path,
    manifest_path: str | Path, designation_path: str | Path,
    selected_scenes_path: str | Path,
    development_registry_paths: Sequence[str | Path],
    development_rows_paths: Sequence[str | Path],
    retired_holdout_paths: Sequence[str | Path], output: str | Path,
    selection_salt_path: str | Path | None = None,
) -> dict[str, Any]:
    singles, development_paths, row_paths, retired_paths = _input_paths(
        actor_path=actor_path, protocol_path=protocol_path,
        program_path=program_path, rcpd_report_path=rcpd_report_path,
        manifest_path=manifest_path, designation_path=designation_path,
        selected_scenes_path=selected_scenes_path,
        development_registry_paths=development_registry_paths,
        development_rows_paths=development_rows_paths,
        retired_holdout_paths=retired_holdout_paths,
    )
    salt_path = _salt_path(selection_salt_path)
    protected = [*singles.values(), *development_paths, *row_paths, *retired_paths,
                 salt_path]
    output_path = _safe_output(output, protected)
    # This preclaim authentication deliberately does not open/hash the program,
    # its report, any final registry, or any producer source.
    identity = _preclaim_identity(singles, development_paths)
    key, registry, started = _claim(identity, output=output_path)
    claim_path = registry / "attempt_started.json"
    claim_sha256 = file_hash(claim_path)
    candidate_identity_sha256 = digest(identity)
    output_created = False
    failure: BaseException | None = None
    holdout_report = audit_report = replay_report = candidate_report = None
    candidate_artifacts: dict[str, str] | None = None
    sources: dict[str, str] = {}
    try:
        output_path.mkdir(parents=True, mode=0o700)
        output_created = True
        _write_exclusive(output_path / "attempt_started.json", _bytes(started))
        # Recheck fixed inputs after the claim, then execute the full strict
        # saved-candidate reader with an authenticated refit before final access.
        if _preclaim_identity(singles, development_paths) != identity:
            raise RuntimeError("Fixed final campaign input changed after claim")
        authenticated_report, sources, developments, pending_artifacts = (
            _authenticate_candidate_after_claim(
            singles=singles, development_paths=development_paths,
            row_paths=row_paths, retired_paths=retired_paths,
        ))
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
            "prior_v7_report_file_sha256": file_hash(
                singles["candidate_prior_v7_report_json"]),
            "prior_v7_rows_file_sha256": file_hash(
                singles["candidate_prior_v7_rows_npz"]),
            "expansion_rows_file_sha256": file_hash(
                singles["candidate_expansion_rows_npz"]),
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
                row_paths=row_paths, identity=identity, key=key,
                claim_sha256=claim_sha256) != pending_artifacts:
            raise RuntimeError(
                "Strictly authenticated candidate artifact lineage changed")
        _write_phase_exclusive(
            registry / "candidate_authenticated.json", _bytes(candidate_marker))
        # Only a successfully returned durable marker establishes this
        # lineage.  A write/fsync failure is recorded separately as a burned
        # phase residue and never masquerades as authenticated evidence.
        candidate_report = authenticated_report
        candidate_artifacts = pending_artifacts
        holdout_dir = output_path / "fresh_holdout"
        holdout_report = holdout_api.build(
            actor_path=singles["actor"],
            protocol_path=singles["protocol"],
            manifest_path=singles["manifest"],
            designation_path=singles["designation"],
            selected_scenes_path=singles["selected_scenes"],
            development_registry_paths=development_paths,
            development_rows_paths=row_paths,
            legacy_v3_rows_path=singles["candidate_prior_v7_rows_npz"],
            retired_holdout_paths=retired_paths,
            output=holdout_dir,
            claim_receipt_path=claim_path,
            expected_claim_sha256=claim_sha256,
            expected_campaign_key=key,
            expected_candidate_identity_sha256=candidate_identity_sha256,
            selection_salt_path=salt_path,
        )
        holdout_api.read_saved_holdout(
            holdout_dir,
            expected_holdout_sha256=file_hash(holdout_dir / "holdout.json"),
            expected_report_sha256=file_hash(holdout_dir / "report.json"),
        )
        holdout_completion = registry / "holdout_completed.json"
        holdout_completion_sha256 = file_hash(holdout_completion)
        expansion_path = developments[holdout_api.DEVELOPMENT_EXPANSION_VERSION][0]
        audit_dir = output_path / "explanation_audit"
        _validate_candidate_artifact_binding(
            _read_json(
                registry / "candidate_authenticated.json",
                "strict candidate authentication phase"),
            singles=singles, row_paths=row_paths)
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
        audit_api.read_saved_report(
            audit_dir,
            expected_report_sha256=file_hash(audit_dir / "report.json"),
            program_path=singles["program"],
            development_rows_paths=row_paths,
            expected_bindings=audit_report["bindings"],
            require_passed=True,
        )
        replay_report = audit_api.replay_saved_audit(
            audit_dir,
            actor_path=singles["actor"],
            protocol_path=singles["protocol"],
            program_path=singles["program"],
            manifest_path=singles["manifest"],
            fresh_holdout_path=holdout_dir / "holdout.json",
            expected_evidence_sha256=file_hash(audit_dir / "evidence.npz"),
            expected_bindings=audit_report["bindings"],
        )
        _validate_candidate_artifact_binding(
            _read_json(
                registry / "candidate_authenticated.json",
                "strict candidate authentication phase"),
            singles=singles, row_paths=row_paths)
        _write_exclusive(output_path / "physical_replay.json", _bytes(replay_report))
        _validate_phase_receipts(
            registry, key=key,
            candidate_identity_sha256=candidate_identity_sha256,
            attempt_started_sha256=claim_sha256, require_complete=True,
            terminal_status="completed_passed")
        if producer_sources() != sources:
            raise RuntimeError("Final-once source closure changed during execution")
    except BaseException as error:
        failure = error

    status = "completed_passed" if failure is None else (
        "burned_failed" if isinstance(failure, Exception) else "burned_interrupted")
    phase_receipts = _phase_receipt_hashes(registry)
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
        "reason": None if failure is None else f"{type(failure).__name__}: {failure}",
        "output_created": output_created,
        "output_identity": str(output_path),
        "candidate_authentication_status": (
            None if candidate_report is None else "passed_strict_reader_and_refit"
        ),
        "candidate_authenticated_sha256": (
            file_hash(registry / "candidate_authenticated.json")
            if candidate_report is not None else None
        ),
        "candidate_artifacts": deepcopy(candidate_artifacts),
        "candidate_artifacts_sha256": (
            None if candidate_artifacts is None else digest(candidate_artifacts)
        ),
        "holdout_status": None if holdout_report is None else holdout_report.get("status"),
        "audit_status": None if audit_report is None else audit_report.get("status"),
        "physical_replay_status": None if replay_report is None else replay_report.get("status"),
        "artifacts": _artifact_hashes(output_path),
        "program_fits": 0,
        "development_authentication_refit": candidate_report is not None,
        "actor_updates": 0,
        "runtime_action_override": False,
        "formal_ready": False,
    }
    completion_raw = _bytes(completion)
    completion_error = None
    try:
        _write_phase_exclusive(registry / "attempt_completed.json", completion_raw)
        if output_created:
            _write_phase_exclusive(output_path / "attempt_completed.json", completion_raw)
    except BaseException as error:
        completion_error = error
    if completion_error is not None:
        raise completion_error
    result = {
        "status": status,
        "key": key,
        "registry": str(registry),
        "output": str(output_path),
        "completion_sha256": (
            file_hash(registry / "attempt_completed.json")
            if (registry / "attempt_completed.json").is_file() else None
        ),
        "reason": completion["reason"],
        "retry_allowed": False,
        "formal_ready": False,
    }
    if failure is not None and not isinstance(failure, Exception):
        raise failure
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
    started_path = _regular(path / "attempt_started.json", "attempt start")
    completion_path = _regular(path / "attempt_completed.json", "attempt completion")
    anchor_path = _regular(_anchor_path(), "permanent final-once anchor")
    if file_hash(completion_path) != expected_completion_sha256:
        raise ValueError("Final-once completion hash differs")
    started = _read_json(started_path, "attempt start")
    completion = _read_json(completion_path, "attempt completion")
    anchor = _read_json(anchor_path, "permanent final-once anchor")
    identity_sha256 = digest(expected_identity)
    phase_receipts = completion.get("phase_receipts")
    actual_phase_receipts = _validate_phase_receipts(
        path, key=key, candidate_identity_sha256=identity_sha256,
        attempt_started_sha256=file_hash(started_path),
        require_complete=completion.get("status") == "completed_passed",
        terminal_status=completion.get("status"),
        ignored_receipts=completion.get("uncommitted_phase_residues", {}))
    if (not isinstance(phase_receipts, Mapping)
            or dict(phase_receipts) != actual_phase_receipts):
        raise ValueError("Final-once phase receipt registry differs")
    candidate_marker_path = path / "candidate_authenticated.json"
    candidate_is_residue = candidate_marker_path.name in completion.get(
        "uncommitted_phase_residues", {})
    if (candidate_marker_path.is_file() and not candidate_marker_path.is_symlink()
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
        raise ValueError("Final-once candidate residue was treated as authenticated")
    if (started.get("key") != key or started.get("campaign_key") != key
            or completion.get("key") != key or completion.get("campaign_key") != key
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
            or completion.get("attempt_started_sha256") != file_hash(started_path)
            or started.get("permanent_anchor_sha256") != file_hash(anchor_path)
            or completion.get("permanent_anchor_sha256") != file_hash(anchor_path)
            or completion.get("automatic_retry") is not False
            or completion.get("retry_allowed") is not False):
        raise ValueError("Final-once permanent record differs")
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
    parser.add_argument("--retired-holdout", action="append", required=True)
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
        retired_holdout_paths=args.retired_holdout, output=args.output,
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
