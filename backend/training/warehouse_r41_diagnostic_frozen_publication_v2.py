"""Strict publication reader for the one frozen r4.1 scene manifest."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Mapping

from backend.training import warehouse_r41_diagnostic_conflict_scenarios as scenes
from backend.training import warehouse_r41_diagnostic_final_once_v8 as final_once
from backend.training import warehouse_r41_diagnostic_frozen_manifest_v2 as manifest_reader
from backend.training.warehouse_native_common import canonical, digest, file_hash
from env.warehouse_native.r41_diagnostic_conflict import diagnostic_contract_receipt


AUTHENTICATION_VERSION = (
    "warehouse-r41-diagnostic-frozen-publication-authentication.v1"
)
SUCCESSFUL_FINAL_VERSION = (
    "warehouse-r41-diagnostic-successful-final-publication-capability.v1"
)
EXPECTED_VALIDATION_SHA256 = (
    "57ea9a2c581a4015df0fd161e14089ebc4f3aa2500d34ae492bd7319f66a9fee"
)
EXPECTED_CONTRACT_SHA256 = (
    "d0eb6c32f30cbdb25b11c267ca2e2685947247ffa1dea8fe96fa338b9cf5f77a"
)
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def _regular(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    if not path.is_file() or path.is_symlink() or path.resolve() != path:
        raise ValueError(label + " must be a canonical regular file")
    return path


def _read(path: Path, label: str) -> dict[str, Any]:
    def pairs(rows):
        value = {}
        for key, item in rows:
            if key in value:
                raise ValueError("Duplicate JSON field in " + label)
            value[key] = item
        return value

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise ValueError(label + " must be one JSON object")
    return value


def _decode_json(raw: bytes, label: str) -> dict[str, Any]:
    def pairs(rows):
        value = {}
        for key, item in rows:
            if key in value:
                raise ValueError("Duplicate JSON field in " + label)
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("Non-finite JSON value in " + label + ": " + token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(label + " must be one JSON object") from error
    if not isinstance(value, dict):
        raise ValueError(label + " must be one JSON object")
    return value


def _hash_regular_fd(path: Path, label: str) -> str:
    """Hash a canonical regular file without a pathname reopen window."""
    canonical_path = _regular(path, label)
    descriptor = os.open(
        canonical_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    digest_value = sha256()
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(label + " must be a canonical regular file")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest_value.update(chunk)
    finally:
        os.close(descriptor)
    return digest_value.hexdigest()


def _successful_final_capability(
    *, actor_path: Path, manifest_path: Path, program_path: Path,
) -> dict[str, Any]:
    """Authenticate the unique successful final chain before final replay."""
    actor_file = _regular(actor_path, "frozen diagnostic Actor")
    manifest_file = _regular(manifest_path, "frozen diagnostic manifest")
    program_file = _regular(program_path, "frozen diagnostic program")
    identity = final_once._campaign_identity()
    campaign_key = final_once.campaign_key()
    registry_root = final_once._ledger_root()
    permanent_anchor = final_once._anchor_path()
    expected_registry_root = Path(
        final_once.holdout_api.DEFAULT_LEDGER_ROOT).expanduser().absolute()
    expected_permanent_anchor = Path(
        final_once.holdout_api.DEFAULT_PERMANENT_ANCHOR).expanduser().absolute()
    account_home = final_once._ACCOUNT_HOME
    if (registry_root != expected_registry_root
            or permanent_anchor != expected_permanent_anchor
            or not registry_root.is_relative_to(account_home)
            or not permanent_anchor.is_relative_to(account_home)
            or registry_root == permanent_anchor
            or registry_root.is_relative_to(permanent_anchor)
            or permanent_anchor.is_relative_to(registry_root)):
        raise ValueError("Final proof must use the fixed account-home ledger and anchor")
    registry = registry_root / campaign_key
    completion_path = registry / "attempt_completed.json"
    if (not completion_path.is_file() or completion_path.is_symlink()
            or completion_path.resolve() != completion_path):
        raise ValueError("Successful final completion is required before final replay")
    completion_sha256 = _hash_regular_fd(
        completion_path, "successful final completion")
    completion = final_once.read_completion(
        registry,
        expected_completion_sha256=completion_sha256,
        expected_identity=identity,
    )
    phase_receipts = completion.get("phase_receipts")
    candidate_artifacts = completion.get("candidate_artifacts")
    if (completion.get("status") != "completed_passed"
            or completion.get("key") != campaign_key
            or completion.get("campaign_key") != campaign_key
            or completion.get("candidate_identity_sha256") != digest(identity)
            or completion.get("output_created") is not True
            or completion.get("reason") is not None
            or completion.get("automatic_retry") is not False
            or completion.get("retry_allowed") is not False
            or completion.get("holdout_status")
                != "passed_program_blind_registry"
            or completion.get("audit_status") != "passed"
            or completion.get("physical_replay_status") != "passed"
            or completion.get("candidate_authentication_status")
                != "passed_strict_reader_and_refit"
            or completion.get("development_authentication_refit") is not True
            or completion.get("program_fits") != 0
            or completion.get("actor_updates") != 0
            or completion.get("runtime_action_override") is not False
            or completion.get("formal_ready") is not False
            or not isinstance(phase_receipts, Mapping)
            or set(phase_receipts) != set(final_once.PHASE_RECEIPT_NAMES)
            or any(_HEX.fullmatch(str(value)) is None
                   for value in phase_receipts.values())
            or completion.get("candidate_authenticated_sha256")
                != phase_receipts.get("candidate_authenticated.json")
            or not isinstance(candidate_artifacts, Mapping)
            or set(candidate_artifacts) != set(final_once.CANDIDATE_ARTIFACT_KEYS)
            or any(_HEX.fullmatch(str(value)) is None
                   for value in candidate_artifacts.values())
            or completion.get("candidate_artifacts_sha256")
                != digest(dict(candidate_artifacts))
            or candidate_artifacts.get("program.json")
                != _hash_regular_fd(program_file, "frozen diagnostic program")
            or _hash_regular_fd(actor_file, "frozen diagnostic Actor")
                != final_once.EXPECTED_ACTOR_SHA256
            or _hash_regular_fd(manifest_file, "frozen diagnostic manifest")
                != final_once.EXPECTED_MANIFEST_SHA256):
        raise ValueError("Exact successful final chain required before final replay")
    return {
        "version": SUCCESSFUL_FINAL_VERSION,
        "status": "authenticated_completed_passed",
        "final_once_version": final_once.VERSION,
        "campaign_key": campaign_key,
        "campaign_identity": deepcopy(identity),
        "campaign_identity_sha256": digest(identity),
        "candidate_identity_sha256": digest(identity),
        "attempt_started_sha256": completion["attempt_started_sha256"],
        "attempt_completed_sha256": completion_sha256,
        "permanent_anchor_sha256": completion["permanent_anchor_sha256"],
        "candidate_authenticated_sha256": completion[
            "candidate_authenticated_sha256"],
        "candidate_artifacts": dict(sorted(candidate_artifacts.items())),
        "candidate_artifacts_sha256": completion[
            "candidate_artifacts_sha256"],
        "phase_receipts": dict(sorted(phase_receipts.items())),
        "phase_receipts_sha256": digest(dict(sorted(phase_receipts.items()))),
        "actor_file_sha256": _hash_regular_fd(
            actor_file, "frozen diagnostic Actor"),
        "manifest_file_sha256": _hash_regular_fd(
            manifest_file, "frozen diagnostic manifest"),
        "program_file_sha256": _hash_regular_fd(
            program_file, "frozen diagnostic program"),
        "retry_allowed": False,
        "runtime_action_override": False,
        "formal_ready": False,
    }


def read_saved_publication(
    *, manifest_path: str | Path, validation_path: str | Path,
    contract_path: str | Path, actor_path: str | Path,
    program_path: str | Path,
) -> dict[str, Any]:
    manifest_file = _regular(manifest_path, "frozen manifest")
    validation_file = _regular(validation_path, "frozen manifest validation")
    contract_file = _regular(contract_path, "frozen diagnostic contract")
    actor_file = _regular(actor_path, "frozen diagnostic Actor")
    program_file = _regular(program_path, "frozen diagnostic program")

    # This capability is the gate: no full-manifest bytes are read or parsed
    # before the non-caller-selectable account-home completion is authenticated.
    successful_final = _successful_final_capability(
        actor_path=actor_file, manifest_path=manifest_file,
        program_path=program_file,
    )

    def snapshot(path: Path, label: str, expected_sha256: str) -> bytes:
        """Read and authenticate parser/loader bytes through one immutable fd."""
        canonical_path = _regular(path, label)
        descriptor = os.open(
            canonical_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        chunks: list[bytes] = []
        actual = sha256()
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(label + " must be a canonical regular file")
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
                actual.update(chunk)
        finally:
            os.close(descriptor)
        if actual.hexdigest() != expected_sha256:
            raise ValueError(label + " bytes differ")
        return b"".join(chunks)

    manifest_raw = snapshot(
        manifest_file, "frozen diagnostic manifest",
        successful_final["manifest_file_sha256"])
    validation_raw = snapshot(
        validation_file, "frozen manifest validation",
        EXPECTED_VALIDATION_SHA256)
    contract_raw = snapshot(
        contract_file, "frozen diagnostic contract", EXPECTED_CONTRACT_SHA256)
    actor_raw = snapshot(
        actor_file, "frozen diagnostic Actor",
        successful_final["actor_file_sha256"])
    # The program is not parsed here, but the same immutable snapshot check
    # prevents its completion binding from being swapped during authentication.
    snapshot(
        program_file, "frozen diagnostic program",
        successful_final["program_file_sha256"])

    manifest = manifest_reader._read_json_object(manifest_raw)
    manifest_content = deepcopy(manifest)
    claimed_manifest_content = manifest_content.pop("content_sha256", None)
    if (claimed_manifest_content
            != manifest_reader.EXPECTED_MANIFEST_CONTENT_SHA256
            or claimed_manifest_content != digest(manifest_content)
            or digest(manifest)
                != manifest_reader.EXPECTED_MANIFEST_SEMANTIC_SHA256):
        raise ValueError("Frozen diagnostic manifest content hash differs")
    manifest_reader._validate_contract(manifest)
    manifest_reader._validate_source_bindings(manifest)
    manifest_reader.runtime_sources()

    actor_temporary = tempfile.TemporaryDirectory(
        prefix="warehouse-r41-publication-actor-")
    actor_snapshot = Path(actor_temporary.name) / "actor.npz"
    descriptor = os.open(
        actor_snapshot,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(actor_raw)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        workload_actor = manifest_reader.load_frozen_actor(actor_snapshot)
        if (workload_actor.artifact_sha256
                != manifest_reader.FROZEN_ACTOR_SHA256
                or workload_actor.metadata.get("actor_parameters_sha256")
                    != manifest_reader.EXPECTED_ACTOR_PARAMETERS_SHA256):
            raise ValueError("Frozen diagnostic Actor parameters differ")
        manifest_reader._validate_rows(
            manifest, workload_actor=workload_actor, replay_scope="all")
    finally:
        actor_temporary.cleanup()

    validation = _decode_json(
        validation_raw, "frozen manifest validation")
    contract = _decode_json(contract_raw, "frozen diagnostic contract")
    if canonical(contract) != canonical(diagnostic_contract_receipt()):
        raise ValueError("Frozen diagnostic contract evidence differs")
    split_counts = {
        name: len(rows) for name, rows in manifest["splits"].items()
    }
    all_rows = [
        row for rows in manifest["splits"].values() for row in rows
    ] + [row for batch in manifest["candidate_batches"] for row in batch]
    workload = validation.get("workload_screen")
    expected_generation = manifest["workload_generation_reports"]
    if (validation.get("version") != scenes.VALIDATION_VERSION
            or validation.get("passed") is not True
            or validation.get("manifest_content_sha256")
                != manifest_reader.EXPECTED_MANIFEST_CONTENT_SHA256
            or validation.get("diagnostic_contract_sha256")
                != contract["contract_sha256"]
            or validation.get("diagnostic_conflict_graph_sha256")
                != manifest["diagnostic_conflict_graph_sha256"]
            or validation.get("conflict_families_sha256")
                != manifest["conflict_families_sha256"]
            or validation.get("producer_sources_sha256")
                != manifest["producer_sources_sha256"]
            or validation.get("base_split_counts") != split_counts
            or validation.get("candidate_batch_count")
                != len(manifest["candidate_batches"])
            or validation.get("candidate_count")
                != sum(map(len, manifest["candidate_batches"]))
            or validation.get("per_family_per_batch")
                != scenes.PER_FAMILY_PER_BATCH
            or validation.get("unique_seed_count")
                != len({row["seed"] for row in all_rows})
            or validation.get("unique_fingerprint_count")
                != len({row["fingerprint"] for row in all_rows})
            or not isinstance(workload, Mapping)
            or workload.get("replayed") is not True
            or workload.get("version")
                != manifest["workload_screen"]["version"]
            or workload.get("contract_sha256")
                != manifest["workload_screen"]["contract_sha256"]
            or workload.get("source_sha256")
                != manifest["workload_screen"]["source_sha256"]
            or workload.get("frozen_actor_sha256")
                != manifest["frozen_actor"]["sha256"]
            or workload.get("generation") != expected_generation
            or validation.get("artifacts") != {
                "diagnostic_contract.json": EXPECTED_CONTRACT_SHA256,
                "manifest.json": successful_final["manifest_file_sha256"],
            }):
        raise ValueError("Frozen diagnostic publication evidence differs")

    current_runtime = manifest_reader.runtime_sources()
    authentication = {
        "version": AUTHENTICATION_VERSION,
        "status": "authenticated_exact_publication_with_full_workload_replay",
        "replay_scope": "all",
        "final_test_replayed_for_authentication": True,
        "final_test_rows_used_for_fit_or_selection": False,
        "final_labels_used": False,
        "final_test_scene_count": len(manifest["splits"]["final_test"]),
        "manifest_file_sha256": successful_final["manifest_file_sha256"],
        "manifest_content_sha256": manifest["content_sha256"],
        "manifest_semantic_sha256": digest(manifest),
        "validation_file_sha256": EXPECTED_VALIDATION_SHA256,
        "validation_semantic_sha256": digest(validation),
        "contract_file_sha256": EXPECTED_CONTRACT_SHA256,
        "contract_semantic_sha256": digest(contract),
        "current_runtime_sources_sha256": current_runtime["sources_sha256"],
        "successful_final_capability_sha256": digest(successful_final),
        "final_once_version": successful_final["final_once_version"],
        "final_once_campaign_key": successful_final["campaign_key"],
        "final_once_campaign_identity_sha256": successful_final[
            "campaign_identity_sha256"],
        "final_once_candidate_identity_sha256": successful_final[
            "candidate_identity_sha256"],
        "final_once_attempt_started_sha256": successful_final[
            "attempt_started_sha256"],
        "final_once_attempt_completed_sha256": successful_final[
            "attempt_completed_sha256"],
        "final_once_permanent_anchor_sha256": successful_final[
            "permanent_anchor_sha256"],
        "final_once_candidate_authenticated_sha256": successful_final[
            "candidate_authenticated_sha256"],
        "final_once_candidate_artifacts_sha256": successful_final[
            "candidate_artifacts_sha256"],
        "final_once_phase_receipts_sha256": successful_final[
                       "phase_receipts_sha256"],
        "actor_file_sha256": successful_final["actor_file_sha256"],
        "program_file_sha256": successful_final["program_file_sha256"],
    }
    if _successful_final_capability(
            actor_path=actor_file, manifest_path=manifest_file,
            program_path=program_file) != successful_final:
        raise RuntimeError(
            "Successful final capability changed during publication authentication")
    return {
        "manifest": deepcopy(manifest),
        "validation": deepcopy(validation),
        "contract": deepcopy(contract),
        "current_runtime": deepcopy(current_runtime),
        "authentication_receipt": deepcopy(authentication),
        "authentication_sha256": digest(authentication),
        "successful_final": deepcopy(successful_final),
        "publication_identity_sha256": digest({
            "manifest": successful_final["manifest_file_sha256"],
            "validation": EXPECTED_VALIDATION_SHA256,
            "contract": EXPECTED_CONTRACT_SHA256,
            "current_runtime_sources_sha256": current_runtime["sources_sha256"],
        }),
    }


__all__ = [
    "AUTHENTICATION_VERSION", "SUCCESSFUL_FINAL_VERSION",
    "EXPECTED_VALIDATION_SHA256",
    "EXPECTED_CONTRACT_SHA256",
    "read_saved_publication",
]
