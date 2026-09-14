"""Fail-closed admission for the warehouse r4.1 diagnostic v11 release.

Only the previously disclosed behaviour-performance gate is waived.  The
the locked public program must pass the irrevocable v13 development outer and
protected final audit before this module can admit any deployment derivative.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any, Mapping

import numpy as np

from backend.training import warehouse_r41_diagnostic_designation_v2 as designation_api
from backend.training import warehouse_r41_diagnostic_explanation_audit_v9 as audit_api
from backend.training import warehouse_r41_diagnostic_final_once_v14 as final_api
from backend.training import warehouse_r41_diagnostic_frozen_manifest_v2 as manifest_api
from backend.training import warehouse_r41_diagnostic_input_snapshot_v8 as snapshot_api
from backend.training import warehouse_r41_diagnostic_final_attempt_closeout_public_v13 as promoted_closeout_api
from backend.training import warehouse_r41_diagnostic_question_bank as question_api
from backend.training import warehouse_r41_diagnostic_rcpd_v13_outer_once as outer_api
from backend.training import warehouse_r41_diagnostic_rcpd_v8 as metrics_api
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend import warehouse_r41_diagnostic_compact_public_tree_v9 as compact_api
from backend.warehouse_r41_diagnostic_online_runtime import (
    PORTABLE_RUNTIME_MANIFEST_VERSION,
    R41DiagnosticOnlineAlignmentRuntime as SourceDiagnosticRuntime,
)
from backend.warehouse_r41_diagnostic_online_runtime_portable_v1 import (
    R41DiagnosticOnlineAlignmentRuntime,
)
from backend.warehouse_r41_diagnostic_public_tree_program_v9 import (
    R41DiagnosticPublicTreeProgramV9,
)
from env.warehouse_native.r41_diagnostic_conflict import (
    CONFLICT_FAMILIES, diagnostic_scene_fingerprint,
)
from ui import warehouse_alignment_r41_diagnostic_release_v9 as release_api
from ui import warehouse_alignment_r41_diagnostic_tutorial as tutorial_api


ROOT = Path(__file__).resolve().parents[2]
VERSION = "warehouse-r41-diagnostic-admission.v11"
STATUS = "admitted_internal_diagnostic_v11"
ARTIFACT_NAMES = (
    "designation", "actor", "protocol", "runtime_protocol", "source_manifest",
    "source_manifest_validation",
    "runtime_manifest",
    "candidate_lock", "development_rows", "promotion_closeout",
    "promotion_identity_registry",
    "promotion_observation_projection", "combined_promoted_projection",
    "promoted_burned_final_rows", "combined_promoted_rows",
    "program", "compact_program", "compact_program_report",
    "outer_result", "outer_collection_report", "outer_rows",
    "final_anchor", "final_completion", "final_material",
    "final_rows", "final_projection_parity", "final_audit",
    "timeout_closeout", "candidate_universe", "timing_calibration",
    "question_bank",
    "question_bank_report", "selected_scenes", "tutorial",
)
GATE_NAMES = (
    "actor_designation", "runtime_action_authority", "portable_manifest",
    "six_high_conflict_scenes", "locked_program",
    "combined_promoted_development_binding", "fresh_outer",
    "protected_final_audit", "v14_final_isolation_inputs",
    "compact_program_parity", "question_bank",
    "neutral_tutorial", "participant_ui_source_closure",
)
MAX_JSON_BYTES = 128 * 1024 * 1024
MAX_BINARY_BYTES = 128 * 1024 * 1024
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_SENSITIVE_KEY = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|secret(?:[_-]?key)?|password|passwd|"
    r"database[_-]?url|authorization|bearer[_-]?token|"
    r"participant[_-]?(?:id|key|record|data))"
    r"(?:$|[_-])", re.I)
_SENSITIVE_VALUE = re.compile(
    r"(?:\bsk-[A-Za-z0-9_-]{16,}|\bBearer\s+[A-Za-z0-9._-]{12,}|"
    r"\b(?:postgres(?:ql)?|mongodb(?:\+srv)?|mysql)://|"
    r"(?:^|[/\\])Users[/\\])", re.I)


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise ValueError("Exact lowercase SHA-256 required for " + label)
    return value


def _content_valid(value: Mapping[str, Any]) -> bool:
    return (type(value.get("content_sha256")) is str
            and value["content_sha256"] == digest({
                key: child for key, child in value.items()
                if key != "content_sha256"
            }))


def _regular(value: str | Path, label: str, *, maximum: int) -> Path:
    path = Path(value).expanduser().absolute()
    stat = path.stat(follow_symlinks=False) if path.exists() else None
    if (stat is None or path.is_symlink() or not path.is_file()
            or path.resolve() != path or not 0 < stat.st_size <= maximum):
        raise ValueError(label + " must be a bounded canonical regular file")
    return path


def _strict_json(path: Path, label: str) -> dict[str, Any]:
    raw = path.read_bytes()

    def pairs(rows):
        value = {}
        for key, child in rows:
            if key in value:
                raise ValueError("Duplicate JSON field in " + label)
            value[key] = child
        return value

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


def _reject_sensitive(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _SENSITIVE_KEY.search(str(key)):
                raise ValueError("Sensitive field in " + label)
            _reject_sensitive(child, label)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_sensitive(child, label)
    elif isinstance(value, str):
        stripped = value.strip()
        if (_SENSITIVE_VALUE.search(value) or Path(stripped).is_absolute()
                or PureWindowsPath(stripped).is_absolute()):
            raise ValueError("Sensitive or absolute-local value in " + label)


def source_closure() -> dict[str, str]:
    """Bind the complete transitive v11 admission implementation."""

    return dict(sorted(local_source_hashes((Path(__file__).resolve(),)).items()))


def package_contract() -> dict[str, Any]:
    """Return the exact deployment transport boundary admitted by v9."""

    return {
        "release_version": release_api.VERSION,
        "public_release_version": release_api.PUBLIC_RELEASE_VERSION,
        "admission_version": VERSION,
        "artifact_paths": deepcopy(release_api.ARTIFACT_PATHS),
        "archive_whitelist": sorted(release_api.ARCHIVE_WHITELIST),
        "maximum_package_bytes": release_api.MAX_PACKAGE_BYTES,
        "maximum_base64_bytes": release_api.MAX_BASE64_BYTES,
        "archive_compression": "ZIP_BZIP2",
        "archive_compresslevel": release_api.ARCHIVE_COMPRESSLEVEL,
        "release_sources_sha256": digest(release_api.release_sources()),
        "protected_outer_or_final_artifacts_packaged": False,
        "runtime_action_override": False,
    }


def _authenticate_official_final_materializer(
    final_anchor: Mapping[str, Any], final_material: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-authenticate the protected-final producer at release admission."""

    source, sources = final_api._official_materializer_binding()
    official = (
        final_api.ROOT / final_api.OFFICIAL_FINAL_MATERIALIZER_RELATIVE_PATH
    ).absolute()
    closure_sha256 = digest(sources)
    anchor_binding = final_anchor.get("bindings")
    material_sources = final_material.get("producer_sources")
    if (source != official
            or closure_sha256
                != final_api.OFFICIAL_FINAL_MATERIALIZER_SOURCE_CLOSURE_SHA256
            or not isinstance(anchor_binding, Mapping)
            or anchor_binding.get(
                "final_materializer_source_closure_sha256")
                != closure_sha256
            or not isinstance(material_sources, Mapping)
            or dict(material_sources) != sources
            or final_material.get("producer_sources_sha256") != closure_sha256):
        raise ValueError("Protected final does not use the official materializer")
    final_api._validate_material(
        final_material, anchor=final_anchor, materializer_sources=sources)
    return {
        "source_relative_path": final_api.OFFICIAL_FINAL_MATERIALIZER_RELATIVE_PATH,
        "source_sha256": sources[
            final_api.OFFICIAL_FINAL_MATERIALIZER_RELATIVE_PATH],
        "source_closure_sha256": closure_sha256,
    }


def _paths(values: Mapping[str, str | Path]) -> dict[str, Path]:
    if not isinstance(values, Mapping) or set(values) != set(ARTIFACT_NAMES):
        raise ValueError("Exact v11 admission artifact set required")
    json_names = set(ARTIFACT_NAMES) - {
        "actor", "compact_program", "development_rows", "outer_rows",
        "final_rows", "combined_promoted_rows", "promoted_burned_final_rows"}
    return {
        name: _regular(values[name], "v9 " + name.replace("_", " "),
                       maximum=(MAX_JSON_BYTES if name in json_names
                                else MAX_BINARY_BYTES))
        for name in ARTIFACT_NAMES
    }


def _portable_scenes(manifest: Mapping[str, Any], runtime: Any) -> list[dict]:
    content = deepcopy(dict(manifest))
    claimed = content.pop("content_sha256", None)
    splits = manifest.get("splits")
    play = splits.get("play") if isinstance(splits, Mapping) else None
    tutorial = splits.get("tutorial") if isinstance(splits, Mapping) else None
    expected_families = {str(row["family_id"]) for row in CONFLICT_FAMILIES}
    if (manifest.get("version") != PORTABLE_RUNTIME_MANIFEST_VERSION
            or claimed != digest(content) or set(splits or {}) != {"play", "tutorial"}
            or not isinstance(play, list) or len(play) != 7
            or not isinstance(tutorial, list) or tutorial != [play[0]]
            or len({row.get("id") for row in play}) != 7
            or len({row.get("fingerprint") for row in play}) != 7
            or {row.get("family_id") for row in play[1:]} != expected_families
            or any(_HEX.fullmatch(str(row.get("fingerprint", ""))) is None
                   for row in play)):
        raise ValueError("Exact tutorial plus six high-conflict scenes required")
    for scene in play:
        environment = runtime.environment(deepcopy(scene))
        if (environment.state.frame != 0
                or diagnostic_scene_fingerprint(environment)
                    != scene["fingerprint"]):
            raise ValueError("Every admitted scene must begin at frame zero")
    return deepcopy(play)


def _designation(value: Mapping[str, Any], files: Mapping[str, Path],
                 lock_bindings: Mapping[str, str]) -> dict[str, Any]:
    bindings = value.get("bindings")
    if (set(value) != designation_api.TOP_LEVEL_FIELDS
            or value.get("version") != designation_api.VERSION
            or value.get("status") != designation_api.STATUS
            or value.get("designated") is not True
            or value.get("behavior_performance_gate_passed") is not False
            or value.get("behavior_performance_gate_waived") is not True
            or value.get("waiver_scope") != ["behavior_performance"]
            or value.get("formal_ready") is not False
            or value.get("formal_sample_eligible") is not False
            or value.get("runtime_action_override") is not False
            or value.get("test_fixture") is not False
            or not isinstance(bindings, Mapping)
            or set(bindings) != designation_api.BINDING_FIELDS
            or bindings.get("actor_sha256") != file_hash(files["actor"])
            or bindings.get("protocol_file_sha256") != file_hash(files["protocol"])
            or bindings.get("actor_sha256") != designation_api.EXPECTED_ACTOR_SHA256
            or lock_bindings.get("designation_sha256")
                != file_hash(files["designation"])):
        raise ValueError("Exact terminal diagnostic Actor designation required")
    return dict(bindings)


def _portable_protocol(source: Mapping[str, Any], portable: Mapping[str, Any],
                       files: Mapping[str, Path]) -> dict[str, Any]:
    runtime_protocol = {
        "version": source.get("version"),
        "feedback": deepcopy(source.get("feedback")),
        "scenario_sampling": deepcopy(source.get("scenario_sampling")),
        "runtime_action_override": source.get("runtime_action_override"),
    }
    content = deepcopy(dict(portable))
    claimed = content.pop("content_sha256", None)
    expected = {
        "version": "warehouse-r41-diagnostic-portable-protocol.v1",
        "source_protocol_file_sha256": file_hash(files["protocol"]),
        "source_protocol_content_sha256": digest(source),
        "runtime_protocol": runtime_protocol,
    }
    if content != expected or claimed != digest(content):
        raise ValueError("Exact public projection of the Actor protocol required")
    _reject_sensitive(portable, "runtime protocol")
    return runtime_protocol


def _compact(files: Mapping[str, Path], *, program_payload: Mapping[str, Any],
             lock_bindings: Mapping[str, str]) -> dict[str, Any]:
    raw_program = files["program"].read_bytes()
    if ((canonical(program_payload) + "\n").encode("utf-8") != raw_program):
        raise ValueError("V9 public program must use canonical JSON bytes")
    program = R41DiagnosticPublicTreeProgramV9.from_dict(program_payload)
    compact_raw = files["compact_program"].read_bytes()
    decoded, header = compact_api.decode_program(
        compact_raw,
        expected_compact_sha256=file_hash(files["compact_program"]),
        expected_source_program_sha256=file_hash(files["program"]),
        expected_actor_feature_names_sha256=lock_bindings[
            "actor_feature_names_sha256"],
        expected_public_feature_contract_sha256=lock_bindings[
            "public_feature_contract_sha256"],
    )
    report = _strict_json(files["compact_program_report"], "compact report")
    transport = report.get("transport")
    audit = report.get("audit")
    audit_sets = audit.get("sets") if isinstance(audit, Mapping) else None
    row_observations = {
        name: outer_api._safe_row_projection(
            files[path_name], expected_sha256=file_hash(files[path_name]),
            fields=frozenset(("observations",)), label=name + " rows",
        )["observations"]
        for name, path_name in (
            ("base development", "development_rows"),
            ("promoted development", "combined_promoted_rows"),
            ("fresh outer", "outer_rows"),
            ("protected final", "final_rows"),
        )
    }
    audited_observations = {
        "development": np.concatenate((
            row_observations["base development"],
            row_observations["promoted development"],
        ), axis=0),
        "fresh_outer": row_observations["fresh outer"],
        "protected_final": row_observations["protected final"],
    }
    _, expected_audit_sets = compact_api._observation_sets(
        audited_observations, expected_features=len(program.base_feature_names))
    if (report.get("version") != compact_api.VERSION
            or report.get("status")
                != "encoded_with_exact_audited_action_parity"
            or not _content_valid(report)
            or report.get("compact_file_sha256")
                != file_hash(files["compact_program"])
            or report.get("source_program_file_sha256")
                != file_hash(files["program"])
            or report.get("source_program_content_sha256") != digest(program_payload)
            or report.get("decoded_program_content_sha256") != digest(decoded.to_dict())
            or not isinstance(audit, Mapping)
            or audit.get("exact_action_parity") is not True
            or type(audit.get("row_count")) is not int or audit["row_count"] <= 0
            or not isinstance(audit_sets, list)
            or audit_sets != expected_audit_sets
            or any(type(row.get("rows")) is not int or row["rows"] <= 0
                   or _HEX.fullmatch(str(row.get("observation_sha256", "")))
                        is None for row in audit_sets)
            or header.get("audit") != audit
            or not isinstance(transport, Mapping)
            or transport.get("fits_secret_file_capacity") is not True
            or report.get("runtime_action_override") is not False
            or report.get("tree_controls_runtime") is not False
            or decoded.action_names != program.action_names
            or digest(decoded.relations.contract())
                != lock_bindings["public_feature_contract_sha256"]):
        raise ValueError("Exact compact v9 program parity evidence required")
    runtime_raw = (canonical(decoded.to_dict()) + "\n").encode("utf-8")
    return {
        "compact_program_sha256": file_hash(files["compact_program"]),
        "compact_program_report_sha256": file_hash(
            files["compact_program_report"]),
        "runtime_program_sha256": sha256(runtime_raw).hexdigest(),
        "runtime_program_content_sha256": digest(decoded.to_dict()),
        "compact_audit_sha256": digest(audit),
        "compact_size": len(compact_raw),
        "parity_rows": audit["row_count"],
    }


def _authenticate_promotion(
    files: Mapping[str, Path], hashes: Mapping[str, str],
    lock_bindings: Mapping[str, str], *, permanent_registry: str | Path,
) -> tuple[dict[str, Any], str]:
    """Authenticate all permanently promoted development rows."""

    if (lock_bindings.get("combined_promoted_rows_sha256")
            != hashes.get("combined_promoted_rows")
            or lock_bindings.get("promotion_closeout_sha256")
                != hashes.get("promotion_closeout")):
        raise ValueError("V13 candidate lock does not bind combined promoted inputs")
    expected_companions = {
        "promotion_closeout": promoted_closeout_api.RECEIPT_NAME,
        "promotion_identity_registry": promoted_closeout_api.IDENTITY_NAME,
        "promotion_observation_projection": (
            promoted_closeout_api.PROMOTED_PROJECTION_NAME),
        "combined_promoted_projection": (
            promoted_closeout_api.COMBINED_PROJECTION_NAME),
        "promoted_burned_final_rows": promoted_closeout_api.PROMOTED_ROWS_NAME,
        "combined_promoted_rows": promoted_closeout_api.COMBINED_ROWS_NAME,
    }
    if (any(files[name].name != filename
            or files[name].parent != files["promotion_closeout"].parent
            for name, filename in expected_companions.items())):
        raise ValueError("Exact v13 promotion-closeout sibling layout required")
    closeout = promoted_closeout_api.read_saved_closeout_public(
        files["promotion_closeout"],
        expected_closeout_sha256=hashes["promotion_closeout"],
        permanent_closeout_registry=permanent_registry,
    )
    consumed = closeout.get("combined_promoted_development")
    semantic = (consumed.get("rows_semantic_sha256")
                if isinstance(consumed, Mapping) else None)
    if (not isinstance(consumed, Mapping)
            or consumed.get("rows_sha256") != hashes["combined_promoted_rows"]
            or type(semantic) is not str or _HEX.fullmatch(semantic) is None):
        raise ValueError(
            "Combined promoted closeout does not bind its development rows")
    return closeout, semantic


def _validate_components_snapshot(
    values: Mapping[str, str | Path], *, outer_permanent_registry: str | Path,
    final_permanent_registry: str | Path,
    permanent_promotion_closeout_registry: str | Path,
) -> dict[str, Any]:
    """Authenticate all release evidence without exposing protected rows."""

    sources = source_closure()
    release_sources = release_api.release_sources()
    files = _paths(values)
    initial_hashes = {name: file_hash(path) for name, path in files.items()}
    json_values = {
        name: _strict_json(path, "v9 " + name.replace("_", " "))
        for name, path in files.items()
        if name not in {
            "actor", "compact_program", "development_rows", "outer_rows",
            "final_rows",
            "combined_promoted_rows", "promoted_burned_final_rows",
        }
    }
    for name in ("runtime_protocol", "runtime_manifest", "question_bank",
                 "tutorial"):
        _reject_sensitive(
            json_values[name], "v11 packaged " + name.replace("_", " "))

    lock_path, lock, lock_bindings = outer_api._candidate_lock(
        files["candidate_lock"],
        expected_sha256=initial_hashes["candidate_lock"])
    if (lock_path != files["candidate_lock"]
            or lock_bindings.get("actor_sha256") != initial_hashes["actor"]
            or lock_bindings.get("protocol_sha256") != initial_hashes["protocol"]
            or lock_bindings.get("runtime_manifest_sha256")
                != initial_hashes["source_manifest"]
            or lock_bindings.get("program_sha256") != initial_hashes["program"]):
        raise ValueError("V9 candidate lock does not bind serving artifacts")
    if (lock_bindings.get("development_rows_sha256")
            != initial_hashes["development_rows"]):
        raise ValueError("V13 candidate lock does not bind development rows")

    promotion_closeout, combined_promoted_rows_semantic_sha256 = (
        _authenticate_promotion(
            files, initial_hashes, lock_bindings,
            permanent_registry=permanent_promotion_closeout_registry))

    designation = _designation(
        json_values["designation"], files, lock_bindings)
    _portable_protocol(
        json_values["protocol"], json_values["runtime_protocol"], files)
    source_manifest_authentication = manifest_api.read_saved_manifest(
        files["source_manifest"],
        expected_sha256=initial_hashes["source_manifest"],
        actor_path=files["actor"], replay_scope="none")
    if (files["source_manifest_validation"]
            != files["source_manifest"].parent / "validation.json"
            or source_manifest_authentication["authentication"].get(
                "validation_file_sha256")
                != initial_hashes["source_manifest_validation"]):
        raise ValueError("Exact frozen source-manifest validation required")
    source_manifest_binding = source_manifest_authentication["authentication"]
    source_manifest = json_values["source_manifest"]
    source_manifest_content = deepcopy(source_manifest)
    source_manifest_claimed = source_manifest_content.pop("content_sha256", None)
    source_runtime = SourceDiagnosticRuntime(
        files["actor"], training_protocol_path=files["protocol"],
        manifest_path=files["source_manifest"],
        expected_actor_sha256=initial_hashes["actor"],
        expected_training_protocol_file_sha256=initial_hashes["protocol"],
        expected_training_protocol_content_sha256=designation[
            "protocol_content_sha256"],
        expected_manifest_file_sha256=initial_hashes["source_manifest"],
        expected_manifest_content_sha256=source_manifest_claimed,
        expected_manifest_semantic_sha256=digest(source_manifest))
    source_runtime.verify_binding()
    manifest = json_values["runtime_manifest"]
    if initial_hashes["source_manifest"] == initial_hashes["runtime_manifest"]:
        raise ValueError("Source and portable runtime manifests must be distinct")
    manifest_content = deepcopy(manifest)
    manifest_claimed = manifest_content.pop("content_sha256", None)
    runtime = R41DiagnosticOnlineAlignmentRuntime(
        files["actor"], training_protocol_path=files["runtime_protocol"],
        manifest_path=files["runtime_manifest"],
        expected_actor_sha256=initial_hashes["actor"],
        expected_training_protocol_file_sha256=initial_hashes["runtime_protocol"],
        expected_training_protocol_content_sha256=designation[
            "protocol_content_sha256"],
        expected_manifest_file_sha256=initial_hashes["runtime_manifest"],
        expected_manifest_content_sha256=manifest_claimed,
        expected_manifest_semantic_sha256=digest(manifest),
    )
    runtime.verify_binding()
    play = _portable_scenes(manifest, runtime)
    if (manifest.get("source_full_manifest_file_sha256")
            != initial_hashes["source_manifest"]
            or manifest.get("source_full_manifest_content_sha256")
                != source_manifest_binding["manifest_content_sha256"]
            or manifest.get("source_full_manifest_semantic_sha256")
                != source_manifest_binding["manifest_semantic_sha256"]):
        raise ValueError("Portable runtime does not bind the frozen source manifest")
    selected = json_values["selected_scenes"]
    selected_play = [selected.get("tutorial"), *(selected.get("X") or ()),
                     *(selected.get("Y") or ())]
    if (selected.get("version")
            != "warehouse-r41-diagnostic-conflict-dynamic-selection.v3"
            or selected.get("status") != "accepted_diagnostic_dynamic_selection"
            or selected.get("release_eligible") is not True
            or selected.get("six_distinct_conflict_families") is not True
            or selected.get("actor_sha256") != initial_hashes["actor"]
            or selected.get("source_manifest_file_sha256")
                != initial_hashes["source_manifest"]
            or selected.get("zero_action_overrides") is not True
            or selected.get("ordinary_sampler_fallback") is not False
            or len(selected_play) != 7
            or [(row.get("id"), row.get("fingerprint"))
                for row in selected_play if isinstance(row, Mapping)]
                != [(row["id"], row["fingerprint"]) for row in play]):
        raise ValueError("Portable scenes do not match the admitted six-scene selection")

    program_payload = json_values["program"]
    compact = _compact(
        files, program_payload=program_payload, lock_bindings=lock_bindings)

    outer = outer_api.read_saved_result(
        files["outer_result"],
        expected_result_sha256=initial_hashes["outer_result"],
        permanent_registry=outer_permanent_registry,
    )
    recomputed_outer_gate = metrics_api._gate(outer.get("metrics", {}))
    if (outer.get("status") != outer_api.STATUS_PASSED
            or outer.get("candidate_lock_sha256")
                != initial_hashes["candidate_lock"]
            or outer.get("program_sha256") != initial_hashes["program"]
            or outer.get("outer_rows_sha256") != initial_hashes["outer_rows"]
            or outer.get("outer_collection_report_sha256")
                != initial_hashes["outer_collection_report"]
            or outer.get("gate", {}).get("passed") is not True
            or outer.get("gate") != recomputed_outer_gate
            or outer.get("row_accounting", {}).get(
                "all_submitted_actions_equal_policy_actions") is not True
            or outer.get("row_accounting", {}).get("runtime_action_overrides") != 0
            or outer.get("execution", {}).get("candidate_refit") is not False
            or outer.get("execution", {}).get("program_mutated") is not False):
        raise ValueError("Passing immutable v9 development outer required")

    completion = final_api.read_completion(
        files["final_completion"],
        expected_completion_sha256=initial_hashes["final_completion"],
        permanent_final_registry=final_permanent_registry,
    )
    final_projection_parity = json_values["final_projection_parity"]
    if (completion.get("status") != final_api.STATUS_PASSED
            or completion.get("final_nine_gates_passed") is not True
            or completion.get("physical_counterfactual_replay_passed") is not True
            or completion.get("whole_scene_and_observation_isolation_passed")
                is not True
            or completion.get(
                "materializer_collector_projection_parity_passed") is not True
            or completion.get("program_fits") != 0
            or completion.get("actor_updates") != 0
            or completion.get("runtime_action_override") is not False
            or completion.get("artifacts", {}).get(final_api.AUDIT_NAME)
                != initial_hashes["final_audit"]
            or completion.get("artifacts", {}).get(final_api.PARITY_NAME)
                != initial_hashes["final_projection_parity"]
            or not _content_valid(final_projection_parity)
            or completion.get("projection_parity_content_sha256")
                != final_projection_parity.get("content_sha256")):
        raise ValueError("Passing immutable v14 protected-final completion required")
    if (files["final_anchor"] != files["final_completion"].parent
            / final_api.ANCHOR_NAME
            or files["final_material"] != files["final_completion"].parent
            / final_api.MATERIAL_NAME
            or files["final_rows"] != files["final_completion"].parent
            / final_api.ROWS_NAME
            or files["final_projection_parity"]
                != files["final_completion"].parent / final_api.PARITY_NAME
            or files["final_audit"] != files["final_completion"].parent
            / final_api.AUDIT_NAME):
        raise ValueError("Exact protected-final artifact sibling layout required")
    final_anchor = json_values["final_anchor"]
    attempt_inputs = final_anchor.get("attempt_key_inputs")
    if (not isinstance(attempt_inputs, Mapping)
            or attempt_inputs.get("candidate_lock_sha256")
                != initial_hashes["candidate_lock"]
            or attempt_inputs.get("actor_sha256") != initial_hashes["actor"]
            or attempt_inputs.get("protocol_sha256") != initial_hashes["protocol"]
            or attempt_inputs.get("runtime_manifest_sha256")
                != initial_hashes["source_manifest"]
            or attempt_inputs.get("designation_sha256")
                != initial_hashes["designation"]
            or attempt_inputs.get("program_sha256") != initial_hashes["program"]
            or attempt_inputs.get("outer_result_sha256")
                != initial_hashes["outer_result"]
            or attempt_inputs.get("timeout_closeout_sha256")
                != initial_hashes["timeout_closeout"]
            or attempt_inputs.get("candidate_universe_sha256")
                != initial_hashes["candidate_universe"]
            or attempt_inputs.get("timing_calibration_sha256")
                != initial_hashes["timing_calibration"]):
        raise ValueError("Protected-final attempt does not bind the admitted candidate")
    timeout_closeout = json_values["timeout_closeout"]
    candidate_universe = json_values["candidate_universe"]
    timing_calibration = json_values["timing_calibration"]
    if (attempt_inputs.get("timeout_closeout_content_sha256")
            != timeout_closeout.get("content_sha256")
            or attempt_inputs.get("candidate_universe_content_sha256")
                != candidate_universe.get("content_sha256")
            or attempt_inputs.get("candidate_universe_identity_sha256")
                != candidate_universe.get("ordered_identity_sha256")
            or attempt_inputs.get("timing_calibration_content_sha256")
                != timing_calibration.get("content_sha256")):
        raise ValueError("Protected-final isolation inputs differ from the anchor")
    final_audit = json_values["final_audit"]
    final_material = json_values["final_material"]
    final_materializer = _authenticate_official_final_materializer(
        final_anchor, final_material)
    expected_audit_bindings = {
        "candidate_lock_sha256": initial_hashes["candidate_lock"],
        "outer_result_sha256": initial_hashes["outer_result"],
        "attempt_anchor_content_sha256": final_anchor["content_sha256"],
        "actor_sha256": initial_hashes["actor"],
        "program_sha256": initial_hashes["program"],
        "public_feature_contract_sha256": lock_bindings[
            "public_feature_contract_sha256"],
        "final_material_file_sha256": initial_hashes["final_material"],
        "final_material_content_sha256": final_material["content_sha256"],
        "final_rows_sha256": initial_hashes["final_rows"],
        "projection_parity_file_sha256": initial_hashes[
            "final_projection_parity"],
        "projection_parity_content_sha256": final_projection_parity[
            "content_sha256"],
    }
    audit_api.validate_report(
        final_audit, expected_bindings=expected_audit_bindings,
        require_passed=True)
    audit_bindings = final_audit["bindings"]
    if (audit_bindings.get("candidate_lock_sha256")
            != initial_hashes["candidate_lock"]
            or audit_bindings.get("outer_result_sha256")
                != initial_hashes["outer_result"]
            or audit_bindings.get("actor_sha256") != initial_hashes["actor"]
            or audit_bindings.get("program_sha256") != initial_hashes["program"]
            or audit_bindings.get("public_feature_contract_sha256")
                != lock_bindings["public_feature_contract_sha256"]):
        raise ValueError("Protected-final audit does not bind the locked candidate")

    question = json_values["question_bank"]
    question_api.validate_payload(source_runtime, question)
    if (question.get("actor_sha256") != initial_hashes["actor"]
            or question.get("protocol_sha256") != runtime.protocol_sha256
            or len(question.get("items", ())) != 8
            or question.get("checks", {}).get("passed") is not True):
        raise ValueError("Exact frozen eight-item question bank required")
    if (files["question_bank"].name != "question_bank.json"
            or files["question_bank_report"].name != "report.json"
            or files["question_bank"].parent
                != files["question_bank_report"].parent):
        raise ValueError("Exact question-bank artifact sibling layout required")
    question_report = question_api.read_saved_report(
        files["question_bank"].parent,
        expected_report_sha256=initial_hashes["question_bank_report"],
        expected_question_bank_sha256=initial_hashes["question_bank"],
        runtime=source_runtime, manifest=source_manifest,
        manifest_file_sha256=initial_hashes["source_manifest"])
    if question_report.get("payload_sha256") != digest(question):
        raise ValueError("Frozen question-bank report differs from its payload")

    tutorial = json_values["tutorial"]
    expected_tutorial_bindings = {
        "scene_manifest_version": manifest["source_full_manifest_version"],
        "scene_manifest_file_sha256": manifest[
            "source_full_manifest_file_sha256"],
        "scene_manifest_content_sha256": manifest[
            "source_full_manifest_content_sha256"],
        "scene_manifest_semantic_sha256": manifest[
            "source_full_manifest_semantic_sha256"],
        "tutorial_scene_fingerprint": play[0]["fingerprint"],
        "tutorial_successor_state_sha256": play[0]["snapshot"]
            ["r41_diagnostic_conflict"]["binding_sha256"],
        "tutorial_snapshot_sha256": digest(play[0]["snapshot"]),
        "diagnostic_contract_sha256": manifest["diagnostic_contract_sha256"],
        "diagnostic_contract_version": manifest["diagnostic_contract_version"],
        "diagnostic_conflict_graph_sha256": manifest[
            "diagnostic_conflict_graph_sha256"],
        "conflict_families_sha256": manifest["conflict_families_sha256"],
        "producer_sources_sha256": digest(tutorial_api.producer_sources()),
    }
    tutorial_replay = tutorial_api.validate_neutral_tutorial(
        tutorial, tutorial_scene=play[0], runtime=source_runtime,
        expected_bindings=expected_tutorial_bindings)

    current_hashes = {name: file_hash(path) for name, path in files.items()}
    if (current_hashes != initial_hashes or source_closure() != sources
            or release_api.release_sources() != release_sources):
        raise RuntimeError("V11 admission input or source changed during validation")
    scene_fingerprints = [str(row["fingerprint"]) for row in play[1:]]
    result = {
        "artifacts": {
            name: {"sha256": initial_hashes[name],
                   "size": files[name].stat(follow_symlinks=False).st_size}
            for name in ARTIFACT_NAMES
        },
        "bindings": {
            "actor_sha256": initial_hashes["actor"],
            "actor_feature_names_sha256": lock_bindings[
                "actor_feature_names_sha256"],
            "protocol_file_sha256": initial_hashes["runtime_protocol"],
            "source_protocol_file_sha256": initial_hashes["protocol"],
            "protocol_content_sha256": runtime.protocol_sha256,
            "runtime_manifest_file_sha256": initial_hashes["runtime_manifest"],
            "runtime_manifest_content_sha256": manifest_claimed,
            "runtime_manifest_semantic_sha256": digest(manifest),
            "runtime_signature": runtime.signature,
            "runtime_manifest_signature": runtime.runtime_manifest_signature,
            "source_manifest_file_sha256": initial_hashes["source_manifest"],
            "source_manifest_content_sha256": source_manifest_binding[
                "manifest_content_sha256"],
            "source_manifest_semantic_sha256": source_manifest_binding[
                "manifest_semantic_sha256"],
            "designation_sha256": initial_hashes["designation"],
            "candidate_lock_sha256": initial_hashes["candidate_lock"],
            "development_rows_sha256": initial_hashes["development_rows"],
            "combined_promoted_rows_sha256": initial_hashes[
                "combined_promoted_rows"],
            "combined_promoted_rows_semantic_sha256": (
                combined_promoted_rows_semantic_sha256),
            "promotion_closeout_sha256": initial_hashes[
                "promotion_closeout"],
            "promotion_closeout_content_sha256": promotion_closeout[
                "content_sha256"],
            "promotion_identity_registry_sha256": initial_hashes[
                "promotion_identity_registry"],
            "promotion_observation_projection_sha256": initial_hashes[
                "promotion_observation_projection"],
            "combined_promoted_projection_sha256": initial_hashes[
                "combined_promoted_projection"],
            "promoted_burned_final_rows_sha256": initial_hashes[
                "promoted_burned_final_rows"],
            "program_sha256": initial_hashes["program"],
            "program_content_sha256": digest(program_payload),
            "public_feature_contract_sha256": lock_bindings[
                "public_feature_contract_sha256"],
            **compact,
            "outer_result_sha256": initial_hashes["outer_result"],
            "outer_result_content_sha256": outer["content_sha256"],
            "outer_collection_report_sha256": initial_hashes[
                "outer_collection_report"],
            "outer_rows_sha256": initial_hashes["outer_rows"],
            "final_completion_sha256": initial_hashes["final_completion"],
            "final_completion_content_sha256": completion["content_sha256"],
            "final_audit_sha256": initial_hashes["final_audit"],
            "final_audit_content_sha256": final_audit["content_sha256"],
            "final_projection_parity_sha256": initial_hashes[
                "final_projection_parity"],
            "final_projection_parity_content_sha256": (
                final_projection_parity["content_sha256"]),
            "final_materializer_source_sha256": final_materializer[
                "source_sha256"],
            "final_materializer_source_closure_sha256": final_materializer[
                "source_closure_sha256"],
            "timeout_closeout_sha256": initial_hashes["timeout_closeout"],
            "timeout_closeout_content_sha256": timeout_closeout[
                "content_sha256"],
            "candidate_universe_sha256": initial_hashes["candidate_universe"],
            "candidate_universe_content_sha256": candidate_universe[
                "content_sha256"],
            "candidate_universe_identity_sha256": candidate_universe[
                "ordered_identity_sha256"],
            "timing_calibration_sha256": initial_hashes["timing_calibration"],
            "timing_calibration_content_sha256": timing_calibration[
                "content_sha256"],
            "play_scene_fingerprints_sha256": digest(
                [str(row["fingerprint"]) for row in play]),
            "formal_scene_fingerprints_sha256": digest(scene_fingerprints),
            "question_bank_sha256": initial_hashes["question_bank"],
            "question_bank_report_sha256": initial_hashes[
                "question_bank_report"],
            "question_bank_content_sha256": digest(question),
            "question_bank_private_items_sha256": question[
                "private_items_sha256"],
            "question_bank_public_items_sha256": question[
                "public_items_sha256"],
            "question_bank_signature": question["source_bank_signature"],
            "tutorial_sha256": initial_hashes["tutorial"],
            "selected_scenes_sha256": initial_hashes["selected_scenes"],
            "tutorial_signature": digest(tutorial),
            "tutorial_replay_sha256": digest(tutorial_replay),
            "release_sources_sha256": digest(release_sources),
            "package_contract_sha256": digest(package_contract()),
        },
        "play_scenes": [{"id": str(row["id"]),
                         "family_id": str(row.get("family_id", "tutorial")),
                         "fingerprint": str(row["fingerprint"]),
                         "scene_sha256": digest(row)} for row in play],
        "sources": {"admission": sources, "release": release_sources},
    }
    _reject_sensitive(result, "v9 admission projection")
    return result


def _snapshot_spec(files: Mapping[str, Path], hashes: Mapping[str, str]):
    unique = sorted(set(files.values()), key=lambda path: path.as_posix())
    parents = {parent: index for index, parent in enumerate(sorted(
        {path.parent for path in unique}, key=lambda path: path.as_posix()))}
    keys = {path: "component%03d" % index
            for index, path in enumerate(unique)}
    snapshot_paths = {keys[path]: path for path in unique}
    expected = {keys[path]: hashes[name] for name, path in files.items()}
    relative = {keys[path]: "groups/%03d/%s" % (
        parents[path.parent], path.name) for path in unique}
    aliases = {name: keys[path] for name, path in files.items()}
    return snapshot_paths, expected, relative, aliases


def validate_components(
    values: Mapping[str, str | Path], *, outer_permanent_registry: str | Path,
    final_permanent_registry: str | Path,
    permanent_promotion_closeout_registry: str | Path,
) -> dict[str, Any]:
    """Authenticate one stable byte snapshot and then recheck originals."""

    sources = source_closure()
    release_sources = release_api.release_sources()
    files = _paths(values)
    hashes = {name: file_hash(path) for name, path in files.items()}
    snapshot_paths, expected, relatives, aliases = _snapshot_spec(files, hashes)
    with snapshot_api.ImmutableInputSnapshot(
            snapshot_paths, expected_sha256=expected,
            relative_names=relatives,
            prefix="warehouse-r41-diagnostic-v11-admission-") as frozen:
        frozen_values = {name: frozen.paths[aliases[name]]
                         for name in ARTIFACT_NAMES}
        checked = _validate_components_snapshot(
            frozen_values, outer_permanent_registry=outer_permanent_registry,
            final_permanent_registry=final_permanent_registry,
            permanent_promotion_closeout_registry=permanent_promotion_closeout_registry)
        frozen.verify()
        if ({name: file_hash(path) for name, path in files.items()} != hashes
                or source_closure() != sources
                or release_api.release_sources() != release_sources):
            raise RuntimeError("V11 admission input or source changed during validation")
        frozen.verify()
        return checked


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().absolute()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (path.exists() or path.is_symlink() or path.parent.is_symlink()
            or path.parent.resolve() != path.parent):
        raise FileExistsError(path)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                         | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write((canonical(value) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def build_admission(
    paths: Mapping[str, str | Path], *, outer_permanent_registry: str | Path,
    final_permanent_registry: str | Path,
    permanent_promotion_closeout_registry: str | Path, output: str | Path,
) -> dict[str, Any]:
    checked = validate_components(
        paths, outer_permanent_registry=outer_permanent_registry,
        final_permanent_registry=final_permanent_registry,
        permanent_promotion_closeout_registry=permanent_promotion_closeout_registry)
    value = {
        "version": VERSION, "status": STATUS,
        "release_version": release_api.PUBLIC_RELEASE_VERSION,
        "pilot_class": release_api.PILOT_CLASS,
        "admitted": True, "behavior_performance_gate_passed": False,
        "behavior_performance_gate_waived": True,
        "waiver_scope": ["behavior_performance"],
        "formal_ready": False, "formal_sample_eligible": False,
        "human_explanation_effect_validated": False,
        "data_persistent": False, "runtime_action_override": False,
        "test_fixture": False,
        "artifacts": checked["artifacts"], "bindings": checked["bindings"],
        "play_scenes": checked["play_scenes"],
        "gates": {name: True for name in GATE_NAMES},
        "sources": checked["sources"],
    }
    value["content_sha256"] = digest(value)
    _reject_sensitive(value, "v9 admission")
    target = Path(output).expanduser().absolute()
    _write_new(target, value)
    expected = file_hash(target)
    try:
        read_saved_admission(
            target, expected_sha256=expected, components=paths,
            outer_permanent_registry=outer_permanent_registry,
            final_permanent_registry=final_permanent_registry,
            permanent_promotion_closeout_registry=permanent_promotion_closeout_registry)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return deepcopy(value)


def read_saved_admission(
    path: str | Path, *, expected_sha256: str,
    components: Mapping[str, str | Path],
    outer_permanent_registry: str | Path,
    final_permanent_registry: str | Path,
    permanent_promotion_closeout_registry: str | Path,
) -> dict[str, Any]:
    saved_path = _regular(path, "v9 diagnostic admission", maximum=MAX_JSON_BYTES)
    if file_hash(saved_path) != _sha(expected_sha256, "v9 admission"):
        raise ValueError("V9 diagnostic admission bytes differ")
    value = _strict_json(saved_path, "v9 diagnostic admission")
    expected_fields = {
        "version", "status", "release_version", "pilot_class", "admitted",
        "behavior_performance_gate_passed",
        "behavior_performance_gate_waived", "waiver_scope", "formal_ready",
        "formal_sample_eligible", "human_explanation_effect_validated",
        "data_persistent", "runtime_action_override", "artifacts", "bindings",
        "test_fixture", "play_scenes", "gates", "sources", "content_sha256",
    }
    if (set(value) != expected_fields or value.get("version") != VERSION
            or value.get("status") != STATUS or value.get("admitted") is not True
            or value.get("release_version") != "r4.1-diagnostic"
            or value.get("pilot_class") != "internal_diagnostic"
            or value.get("behavior_performance_gate_passed") is not False
            or value.get("behavior_performance_gate_waived") is not True
            or value.get("waiver_scope") != ["behavior_performance"]
            or any(value.get(key) is not False for key in (
                "formal_ready", "formal_sample_eligible",
                "human_explanation_effect_validated", "data_persistent",
                "runtime_action_override", "test_fixture"))
            or value.get("gates") != {name: True for name in GATE_NAMES}
            or not _content_valid(value)):
        raise ValueError("Exact v9 internal diagnostic admission required")
    checked = validate_components(
        components, outer_permanent_registry=outer_permanent_registry,
        final_permanent_registry=final_permanent_registry,
        permanent_promotion_closeout_registry=permanent_promotion_closeout_registry)
    for name in ("artifacts", "bindings", "play_scenes", "sources"):
        if value.get(name) != checked[name]:
            raise ValueError("V11 admission " + name + " differs from evidence")
    _reject_sensitive(value, "saved v11 admission")
    return deepcopy(value)


__all__ = [
    "VERSION", "STATUS", "ARTIFACT_NAMES", "GATE_NAMES",
    "source_closure", "package_contract", "validate_components", "build_admission",
    "read_saved_admission",
]
