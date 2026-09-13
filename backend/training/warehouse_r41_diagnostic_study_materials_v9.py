"""Freeze the v9 diagnostic questionnaire and neutral tutorial.

This producer is deliberately downstream of the irrevocable v9 final audit.
It reads no outer/final rows and no holdout salt.  It authenticates the locked
Actor and public program, requires the saved final explanation audit to pass,
then reuses the established participant-facing questionnaire and neutral
AI--AI tutorial generators.  Their public schemas consequently remain
compatible with the existing study UI.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping

from backend.training import warehouse_r41_diagnostic_explanation_audit_v9 as audit_api
from backend.training import warehouse_r41_diagnostic_question_bank as question_api
from backend.training import warehouse_r41_diagnostic_rcpd_v9_outer_once as outer_api
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.warehouse_r41_diagnostic_online_runtime import (
    R41DiagnosticOnlineAlignmentRuntime,
)
from backend.warehouse_r41_diagnostic_public_tree_program_v9 import (
    R41DiagnosticPublicTreeProgramV9,
)
from ui import warehouse_alignment_r41_diagnostic_tutorial as tutorial_api


VERSION = "warehouse-r41-diagnostic-study-materials.v9"
STATUS = "frozen_after_passed_v9_final_audit"
QUESTION_DIRECTORY = "question_bank"
QUESTION_FILE = "question_bank.json"
QUESTION_REPORT = "report.json"
TUTORIAL_FILE = "tutorial.json"
RECEIPT_FILE = "receipt.json"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_FINAL_BINDINGS = frozenset((
    "candidate_lock_sha256", "outer_result_sha256",
    "attempt_anchor_content_sha256", "actor_sha256", "program_sha256",
    "public_feature_contract_sha256", "final_material_file_sha256",
    "final_material_content_sha256", "final_rows_sha256",
))


def producer_sources() -> dict[str, str]:
    return dict(sorted(local_source_hashes((Path(__file__).resolve(),)).items()))


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise ValueError("Exact lowercase SHA-256 required for " + label)
    return value


def _regular(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or not 0 < path.stat(follow_symlinks=False).st_size <= 512 * 1024 * 1024):
        raise ValueError(label + " must be a bounded canonical regular file")
    return path


def _read_json(value: str | Path, label: str, *, expected_sha256: str) -> tuple[Path, dict]:
    path = _regular(value, label)
    if file_hash(path) != _sha(expected_sha256, label + " SHA-256"):
        raise ValueError("Exact " + label + " bytes required")

    def pairs(items):
        result = {}
        for key, child in items:
            if key in result:
                raise ValueError("Duplicate JSON field in " + label)
            result[key] = child
        return result

    try:
        parsed = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("Non-finite JSON value in " + label + ": " + token)),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(label + " must be strict UTF-8 JSON") from error
    if not isinstance(parsed, dict):
        raise ValueError(label + " must contain one JSON object")
    return path, parsed


def _runtime(actor: Path, protocol: Path, manifest_path: Path,
             manifest: Mapping[str, Any]) -> R41DiagnosticOnlineAlignmentRuntime:
    protocol_value = json.loads(protocol.read_text(encoding="utf-8"))
    content = deepcopy(dict(manifest))
    claimed = content.pop("content_sha256", None)
    if claimed != digest(content):
        raise ValueError("Diagnostic source manifest content differs")
    return R41DiagnosticOnlineAlignmentRuntime(
        actor, training_protocol_path=protocol, manifest_path=manifest_path,
        expected_actor_sha256=file_hash(actor),
        expected_training_protocol_file_sha256=file_hash(protocol),
        expected_training_protocol_content_sha256=digest(protocol_value),
        expected_manifest_file_sha256=file_hash(manifest_path),
        expected_manifest_content_sha256=claimed,
        expected_manifest_semantic_sha256=digest(manifest),
    )


def authenticate_release_inputs(
    *, actor: str | Path, expected_actor_sha256: str,
    protocol: str | Path, expected_protocol_sha256: str,
    source_manifest: str | Path, expected_source_manifest_sha256: str,
    candidate_lock: str | Path, expected_candidate_lock_sha256: str,
    program: str | Path, expected_program_sha256: str,
    final_audit: str | Path, expected_final_audit_sha256: str,
) -> dict[str, Any]:
    """Authenticate the complete prerequisite boundary without holdout access."""

    actor_path = _regular(actor, "frozen v9 Actor")
    protocol_path = _regular(protocol, "Actor training protocol")
    if file_hash(actor_path) != _sha(expected_actor_sha256, "Actor"):
        raise ValueError("Exact frozen v9 Actor bytes required")
    if file_hash(protocol_path) != _sha(expected_protocol_sha256, "protocol"):
        raise ValueError("Exact Actor training protocol bytes required")
    manifest_path, manifest = _read_json(
        source_manifest, "source manifest",
        expected_sha256=expected_source_manifest_sha256)
    lock_path = _regular(candidate_lock, "v9 candidate lock")
    lock_path, lock, lock_bindings = outer_api._candidate_lock(
        lock_path, expected_sha256=expected_candidate_lock_sha256)
    program_path, program_payload = _read_json(
        program, "locked v9 program", expected_sha256=expected_program_sha256)
    audit_path, audit = _read_json(
        final_audit, "passed v9 final audit",
        expected_sha256=expected_final_audit_sha256)

    expected_lock = {
        "actor_sha256": file_hash(actor_path),
        "protocol_sha256": file_hash(protocol_path),
        "runtime_manifest_sha256": file_hash(manifest_path),
        "program_sha256": file_hash(program_path),
    }
    if any(lock_bindings.get(name) != value
           for name, value in expected_lock.items()):
        raise ValueError("Candidate lock does not bind the study-material inputs")
    parsed_program = R41DiagnosticPublicTreeProgramV9.from_dict(program_payload)
    runtime = _runtime(actor_path, protocol_path, manifest_path, manifest)
    runtime.verify_binding()
    if (tuple(parsed_program.action_names)
            != tuple(runtime.actor.metadata.get("actions", ()))
            or tuple(parsed_program.base_feature_names)
            != tuple(runtime.actor.metadata.get("feature_names", ()))
            or digest(parsed_program.relations.contract())
            != lock_bindings.get("public_feature_contract_sha256")):
        raise ValueError("Locked v9 program and frozen Actor registry differ")

    final_bindings = audit.get("bindings")
    if (not isinstance(final_bindings, Mapping)
            or set(final_bindings) != _FINAL_BINDINGS
            or final_bindings.get("candidate_lock_sha256") != file_hash(lock_path)
            or final_bindings.get("actor_sha256") != file_hash(actor_path)
            or final_bindings.get("program_sha256") != file_hash(program_path)
            or final_bindings.get("public_feature_contract_sha256")
                != lock_bindings.get("public_feature_contract_sha256")):
        raise ValueError("Final audit does not bind the locked v9 candidate")
    audit_api.validate_report(
        audit, expected_bindings=dict(final_bindings), require_passed=True)

    initial_hashes = {
        "actor_sha256": file_hash(actor_path),
        "protocol_sha256": file_hash(protocol_path),
        "source_manifest_sha256": file_hash(manifest_path),
        "candidate_lock_sha256": file_hash(lock_path),
        "program_sha256": file_hash(program_path),
        "final_audit_sha256": file_hash(audit_path),
    }
    return {
        "paths": {
            "actor": actor_path, "protocol": protocol_path,
            "source_manifest": manifest_path, "candidate_lock": lock_path,
            "program": program_path, "final_audit": audit_path,
        },
        "hashes": initial_hashes,
        "runtime": runtime,
        "manifest": manifest,
        "lock": lock,
        "lock_bindings": lock_bindings,
        "program": parsed_program,
        "program_payload": program_payload,
        "final_audit": audit,
    }


def _validate_question_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    items = payload.get("items") if isinstance(payload, Mapping) else None
    if not isinstance(items, list) or len(items) != 8:
        raise ValueError("V9 study question bank requires exactly eight items")
    kinds = {"next_action": 0, "wait_three": 0}
    scenes = set()
    for item in items:
        kind = item.get("kind") if isinstance(item, Mapping) else None
        if kind not in kinds:
            raise ValueError("V9 study question kind differs")
        kinds[kind] += 1
        prompt = item.get("prompt")
        options = item.get("options")
        snapshot = item.get("snapshot")
        if (not isinstance(prompt, Mapping) or set(prompt) != {"zh", "en"}
                or any(type(prompt[name]) is not str or not prompt[name].strip()
                       for name in ("zh", "en"))
                or not isinstance(options, list)
                or item.get("answer") not in {row.get("value") for row in options
                                               if isinstance(row, Mapping)}
                or not isinstance(snapshot, Mapping)
                or digest(snapshot) != item.get("snapshot_sha256")
                or snapshot.get("state", {}).get("frame") != item.get("frame")):
            raise ValueError("V9 study question public/frame binding differs")
        scenes.add(item.get("scenario_id"))
        if kind == "wait_three":
            evidence = item.get("evidence")
            if (not isinstance(evidence, Mapping)
                    or evidence.get("assumed_player_actions")
                        != ["WAIT", "WAIT", "WAIT"]
                    or not isinstance(evidence.get("transitions"), list)
                    or len(evidence["transitions"]) != 3
                    or evidence.get("transitions_sha256")
                        != digest(evidence["transitions"])):
                raise ValueError("V9 three-step counterfactual binding differs")
    if kinds != {"next_action": 4, "wait_three": 4} or len(scenes) != 8:
        raise ValueError("V9 study question balance or scene isolation differs")
    return {
        "bilingual_items": 8,
        "next_action_items_replayed": 4,
        "wait_three_items_replayed": 4,
        "independent_source_scenes": 8,
    }


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    raw = (canonical(value) + "\n").encode("utf-8")
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def build(*, output: str | Path, **inputs: Any) -> dict[str, Any]:
    """Build all participant study materials into one atomic directory."""

    output_path = Path(output).expanduser().absolute()
    if (output_path.exists() or output_path.is_symlink()
            or output_path.parent.is_symlink()):
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.parent.resolve() != output_path.parent.absolute():
        raise ValueError("V9 study-material output parent is unsafe")

    authenticated = authenticate_release_inputs(**inputs)
    runtime = authenticated["runtime"]
    manifest = authenticated["manifest"]
    manifest_sha = authenticated["hashes"]["source_manifest_sha256"]
    staging = Path(tempfile.mkdtemp(
        prefix="." + output_path.name + ".", dir=output_path.parent))
    staging.chmod(0o700)
    try:
        question_dir = staging / QUESTION_DIRECTORY
        question_report = question_api.build(
            runtime, manifest, output=question_dir,
            manifest_file_sha256=manifest_sha)
        question_path = question_dir / QUESTION_FILE
        question_report_path = question_dir / QUESTION_REPORT
        question_payload = json.loads(question_path.read_text(encoding="utf-8"))
        question_api.validate_payload(runtime, question_payload)
        question_contract = _validate_question_contract(question_payload)
        replayed_report = question_api.read_saved_report(
            question_dir,
            expected_report_sha256=file_hash(question_report_path),
            expected_question_bank_sha256=file_hash(question_path),
            runtime=runtime, manifest=manifest,
            manifest_file_sha256=manifest_sha)
        if digest(question_report) != digest(replayed_report):
            raise ValueError("V9 question-bank report replay differs")

        tutorial_payload = tutorial_api.build_neutral_tutorial(
            manifest, manifest_file_sha256=manifest_sha)
        tutorial_scene = manifest.get("splits", {}).get("tutorial", [None])[0]
        if not isinstance(tutorial_scene, Mapping):
            raise ValueError("V9 neutral tutorial scene is missing")
        tutorial_replay = tutorial_api.validate_neutral_tutorial(
            tutorial_payload, tutorial_scene=tutorial_scene, runtime=runtime,
            expected_bindings=tutorial_api.bindings(
                manifest, tutorial_scene, manifest_file_sha256=manifest_sha))
        if (tutorial_payload.get("uses_final_actor") is not False
                or tutorial_payload.get("source") != tutorial_api.SOURCE
                or tutorial_replay.get("passed") is not True):
            raise ValueError("V9 tutorial must be neutral and physically replayable")
        tutorial_path = staging / TUTORIAL_FILE
        _write_exclusive(tutorial_path, tutorial_payload)

        current = {name + "_sha256": file_hash(path)
                   for name, path in authenticated["paths"].items()}
        if current != authenticated["hashes"]:
            raise RuntimeError("V9 study-material input changed during generation")
        final_audit = authenticated["final_audit"]
        sources = producer_sources()
        receipt: dict[str, Any] = {
            "version": VERSION,
            "status": STATUS,
            "bindings": {
                **authenticated["hashes"],
                "candidate_lock_content_sha256": authenticated["lock"][
                    "content_sha256"],
                "program_content_sha256": digest(
                    authenticated["program_payload"]),
                "public_feature_contract_sha256": authenticated[
                    "lock_bindings"]["public_feature_contract_sha256"],
                "final_audit_content_sha256": final_audit["content_sha256"],
                "question_bank_sha256": file_hash(question_path),
                "question_bank_report_sha256": file_hash(question_report_path),
                "tutorial_sha256": file_hash(tutorial_path),
            },
            "question_bank": {
                **question_contract,
                "actual_actor_actions_independently_replayed": True,
                "counterfactuals_do_not_mutate_source_frames": True,
                "participant_projection_schema_unchanged": True,
            },
            "tutorial": {
                "source": tutorial_api.SOURCE,
                "uses_final_actor": False,
                "rules_only": True,
                "physical_replay_passed": True,
                "frame_count": tutorial_replay["frame_count"],
                "duration_ms": tutorial_payload["duration_ms"],
            },
            "information_boundary": {
                "outer_rows_read": False,
                "final_rows_read": False,
                "holdout_salt_read": False,
                "program_controls_runtime_actions": False,
                "answer_access_delegated_to_existing_v9_online_explanation": True,
            },
            "producer_sources": sources,
            "producer_sources_sha256": digest(sources),
            "formal_ready": False,
        }
        receipt["content_sha256"] = digest(receipt)
        _write_exclusive(staging / RECEIPT_FILE, receipt)
        os.replace(staging, output_path)
        return receipt
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "VERSION", "STATUS", "QUESTION_DIRECTORY", "QUESTION_FILE",
    "QUESTION_REPORT", "TUTORIAL_FILE", "RECEIPT_FILE", "producer_sources",
    "authenticate_release_inputs", "build",
]
