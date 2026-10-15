"""Irrevocably close and promote the burned v12 protected-final attempt.

The protected-final attempt authenticated and permanently claimed the frozen
v12 candidate before its 64 whole-scene identities were materialised.  It
then completed as ``burned_failed``.  This append-only closeout authenticates
that exact claim/completion/material pair before it accesses any scene, and
only then deterministically replays those now-development-exposed scenes with
the same frozen neural Actor.

The closeout publishes three development sources in a fixed order: the
already promoted v11 outer, the consumed/passed v12 outer, and the post-closeout
recollection of the burned v12 final.  It never invokes the protected
materializer, never accepts a secret configuration, and can never retry the
same protected-final attempt.
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
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from backend.training import warehouse_r41_diagnostic_explanation_audit_v9 as audit_api
from backend.training import warehouse_r41_diagnostic_final_once_v9 as final_api
from backend.training import warehouse_r41_diagnostic_outer_attempt_closeout_v12 as v11_closeout_api
from backend.training import warehouse_r41_diagnostic_outer_collection_v12 as collection_api
from backend.training import warehouse_r41_diagnostic_outer_hash_projection_v12 as projection_api
from backend.training import warehouse_r41_diagnostic_rcpd_v7 as rows_api
from backend.training import warehouse_r41_diagnostic_rcpd_v8 as metrics_api
from backend.training import warehouse_r41_diagnostic_rcpd_v12_outer_once as outer_api
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.warehouse_r41_diagnostic_public_tree_program_v9 import (
    R41DiagnosticPublicTreeProgramV9,
)


VERSION = "warehouse-r41-diagnostic-final-attempt-closeout.v13"
STATUS = "burned_v12_final_irrevocably_closed_and_promoted"
RECEIPT_NAME = "closeout_receipt.json"
IDENTITY_NAME = "burned_identity_registry.json"
PROMOTED_PROJECTION_NAME = "ordered_promoted_observation_hashes.json"
COMBINED_PROJECTION_NAME = "combined_promoted_observation_hashes.json"
PROMOTED_ROWS_NAME = "promoted_rows.npz"
COMBINED_ROWS_NAME = "combined_promoted_rows.npz"
FINAL_SCENE_COUNT = 64
FAMILY_COUNTS = {
    "conflict_family_01": 11,
    "conflict_family_02": 11,
    "conflict_family_03": 11,
    "conflict_family_04": 11,
    "conflict_family_05": 10,
    "conflict_family_06": 10,
}
MAX_JSON_BYTES = 512 * 1024 * 1024
MAX_NPZ_BYTES = 2 * 1024 * 1024 * 1024
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def producer_sources() -> dict[str, str]:
    return dict(sorted(local_source_hashes((Path(__file__).resolve(),)).items()))


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "authenticate_burned_completion_before_scene_access": True,
        "same_final_attempt_retry_permitted": False,
        "protected_salt_reused": False,
        "protected_materializer_invoked": False,
        "post_closeout_deterministic_development_recollection": True,
        "independent_physical_replay": True,
        "runtime_action_override": False,
        "program_controls_runtime_actions": False,
        "combined_promoted_source_order": [
            "consumed_v11_outer", "consumed_v12_outer", "burned_v12_final",
        ],
        "all_promoted_rows_marked_development": True,
        "burned_final_identity_and_observation_reuse_permitted": False,
        "consumed_v12_outer_identity_and_observation_reuse_permitted": False,
        "formal_ready": False,
    }


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


def _directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    if not path.is_dir() or path.is_symlink() or path.resolve() != path:
        raise ValueError(label + " must be a canonical directory")
    return path


def _regular(value: str | Path, label: str, *, maximum: int,
             expected_sha256: str | None = None) -> Path:
    path = Path(value).expanduser().absolute()
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or path.stat(follow_symlinks=False).st_size <= 0
            or path.stat(follow_symlinks=False).st_size > maximum):
        raise ValueError(label + " must be a bounded canonical regular file")
    if expected_sha256 is not None and file_hash(path) != _sha(
            expected_sha256, label + " SHA-256"):
        raise ValueError("Exact " + label + " bytes required")
    return path


def _strict_json(value: str | Path, label: str, *, expected_sha256: str
                 ) -> tuple[Path, bytes, dict[str, Any]]:
    path = _regular(value, label, maximum=MAX_JSON_BYTES,
                    expected_sha256=expected_sha256)
    raw = path.read_bytes()

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in items:
            if key in result:
                raise ValueError("Duplicate JSON field in " + label)
            result[key] = child
        return result

    try:
        parsed = json.loads(
            raw.decode("utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("Non-finite JSON value in " + label + ": " + token)),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(label + " must be strict UTF-8 JSON") from error
    if not isinstance(parsed, dict) or file_hash(path) != expected_sha256:
        raise RuntimeError(label + " changed during read")
    return path, raw, parsed


def _decode_hashes(array: np.ndarray, label: str) -> list[str]:
    if array.ndim != 1 or array.dtype.kind != "S":
        raise ValueError(label + " must be a one-dimensional byte-string vector")
    try:
        result = list(map(str, np.char.decode(array, "ascii")))
    except UnicodeDecodeError as error:
        raise ValueError(label + " must be ASCII") from error
    if any(_HEX.fullmatch(value) is None for value in result):
        raise ValueError(label + " entries must be SHA-256 values")
    return result


def _arrays_digest(arrays: Mapping[str, np.ndarray]) -> str:
    summary: dict[str, Any] = {}
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        summary[name] = {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": sha256(memoryview(value).cast("B")).hexdigest(),
        }
    return digest(summary)


def _combine_rows(*sources: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Concatenate exact row schemas in caller order and mark development."""
    if not sources:
        raise ValueError("At least one promoted row source is required")
    expected = set(rows_api._FIELDS)
    for index, arrays in enumerate(sources):
        if set(arrays) != expected:
            raise ValueError(f"Promoted source {index} row schema differs")
        count = len(arrays["split_validation"])
        for name in expected:
            if arrays[name].ndim < 1 or len(arrays[name]) != count:
                raise ValueError(f"Promoted source {index} row alignment differs")
        if not np.all(arrays["split_validation"]):
            raise ValueError(f"Promoted source {index} contains non-development rows")
    first = sources[0]
    for index, arrays in enumerate(sources[1:], start=1):
        for name in expected:
            if (arrays[name].dtype != first[name].dtype
                    or arrays[name].shape[1:] != first[name].shape[1:]):
                raise ValueError(
                    f"Promoted source {index} dtype or trailing shape differs at {name}")
    result = {
        name: np.ascontiguousarray(np.concatenate(
            [arrays[name] for arrays in sources], axis=0))
        for name in sorted(expected)
    }
    result["split_validation"] = np.ones(
        len(result["split_validation"]), dtype=np.bool_)
    return result


def _public_identity_registry(
    scenes: Sequence[Mapping[str, Any]], *, expected_count: int = FINAL_SCENE_COUNT,
) -> dict[str, Any]:
    identities: list[dict[str, Any]] = []
    for scene in scenes:
        if not isinstance(scene, Mapping):
            raise ValueError("Burned-final scene identity differs")
        identity = {name: scene.get(name) for name in (
            "batch_index", "family_id", "seed", "fingerprint")}
        if (type(identity["batch_index"]) is not int
                or identity["batch_index"] < 0
                or type(identity["family_id"]) is not str
                or not identity["family_id"]
                or type(identity["seed"]) is not int
                or isinstance(identity["seed"], bool)
                or identity["seed"] < 0
                or type(identity["fingerprint"]) is not str
                or _HEX.fullmatch(identity["fingerprint"]) is None):
            raise ValueError("Burned-final public identity differs")
        identities.append(identity)
    if (len(identities) != expected_count
            or len({row["seed"] for row in identities}) != expected_count
            or len({row["fingerprint"] for row in identities}) != expected_count):
        raise ValueError("Burned-final public identity population differs")
    family_counts = dict(sorted(Counter(
        row["family_id"] for row in identities).items()))
    value: dict[str, Any] = {
        "version": VERSION + ".burned-identity-registry.v1",
        "status": "burned_final_identities_permanently_excluded",
        "scene_count": len(identities),
        "identities": identities,
        "identities_sha256": digest(identities),
        "family_counts": family_counts,
        "scene_snapshots_included": False,
        "rng_state_included": False,
        "hidden_selection_material_included": False,
        "formal_ready": False,
    }
    value["content_sha256"] = digest(value)
    return value


def _observation_projection(
    arrays: Mapping[str, np.ndarray], *, version: str, source: str,
) -> dict[str, Any]:
    ordered = _decode_hashes(arrays["observation_hashes"],
                             source + " observation hashes")
    unique = sorted(set(ordered))
    value: dict[str, Any] = {
        "version": version,
        "status": "development_exposed_one_way_observation_hash_projection",
        "source": source,
        "row_count": len(ordered),
        "ordered_observation_hashes": ordered,
        "ordered_observation_hashes_sha256": digest(ordered),
        "unique_outer_observation_hash_count": len(unique),
        "outer_observation_hashes": unique,
        "outer_observation_hashes_sha256": digest(unique),
        "raw_observation_values_included": False,
        "action_values_included": False,
        "probability_values_included": False,
        "label_values_included": False,
        "formal_ready": False,
    }
    value["content_sha256"] = digest(value)
    return value


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


def _strict_json_object(path: str | Path, label: str,
                        expected_sha256: str) -> tuple[Path, dict[str, Any]]:
    file_path, _, value = _strict_json(
        path, label, expected_sha256=expected_sha256)
    if not _content_valid(value):
        raise ValueError(label + " content digest differs")
    return file_path, value


def _authenticate_burned_inputs(
    *, final_anchor_path: str | Path, expected_final_anchor_sha256: str,
    final_completion_path: str | Path, expected_final_completion_sha256: str,
    materializer_output_path: str | Path,
    expected_materializer_output_sha256: str,
    permanent_final_registry: str | Path,
    candidate_lock_path: str | Path, expected_candidate_lock_sha256: str,
    actor_path: str | Path, protocol_path: str | Path,
    runtime_manifest_path: str | Path, program_path: str | Path,
    selector_report_path: str | Path,
    v12_outer_registry_path: str | Path,
    v12_outer_registry_report_path: str | Path,
    v12_outer_projection_path: str | Path,
    v12_outer_projection_receipt_path: str | Path,
    v12_outer_collection_rows_path: str | Path,
    v12_outer_collection_report_path: str | Path,
    v12_outer_result_path: str | Path, expected_v12_outer_result_sha256: str,
    permanent_v12_outer_registry: str | Path,
    v11_closeout_path: str | Path, expected_v11_closeout_sha256: str,
    permanent_v11_outer_registry: str | Path,
    v11_rows_path: str | Path,
) -> dict[str, Any]:
    """Authenticate the burned claim and every source before scene access."""
    anchor_path, _, anchor = _strict_json(
        final_anchor_path, "burned v12 final anchor",
        expected_sha256=expected_final_anchor_sha256)
    completion_path, _, completion = _strict_json(
        final_completion_path, "burned v12 final completion",
        expected_sha256=expected_final_completion_sha256)
    material_path, _, material = _strict_json(
        materializer_output_path, "burned v12 claim-local materializer output",
        expected_sha256=expected_materializer_output_sha256)

    attempt_key = anchor.get("attempt_key")
    if (anchor.get("version") != final_api.VERSION + ".attempt-anchor.v1"
            or anchor.get("status") != "final_attempt_irrevocably_claimed"
            or type(attempt_key) is not str or _HEX.fullmatch(attempt_key) is None
            or not _content_valid(anchor)
            or anchor.get("candidate_and_outer_authenticated_before_claim") is not True
            or anchor.get("final_identity_or_rows_accessed_before_claim") is not False
            or anchor.get("retry_allowed") is not False
            or anchor.get("formal_ready") is not False):
        raise ValueError("Burned v12 final anchor semantics differ")
    if (set(completion) != final_api._FAILURE_COMPLETION_FIELDS
            or completion.get("version") != final_api.VERSION
            or completion.get("status") != final_api.STATUS_FAILED
            or completion.get("reason") != "protected_final_phase_failed"
            or completion.get("attempt_key") != attempt_key
            or completion.get("attempt_anchor_content_sha256")
                != anchor["content_sha256"]
            or completion.get("final_consumed") is not True
            or completion.get("retry_allowed") is not False
            or completion.get("program_fits") != 0
            or completion.get("actor_updates") != 0
            or completion.get("runtime_action_override") is not False
            or completion.get("formal_ready") is not False
            or not _content_valid(completion)):
        raise ValueError("Burned v12 final completion semantics differ")

    permanent = _directory(permanent_final_registry,
                           "permanent burned-final registry")
    campaign = _directory(permanent / attempt_key,
                          "permanent burned-final campaign")
    expected_entries = {
        final_api.ANCHOR_NAME, final_api.COMPLETION_NAME, "materializer_output.json",
    }
    if ({entry.name for entry in campaign.iterdir()} != expected_entries
            or anchor_path != (campaign / final_api.ANCHOR_NAME).absolute()
            or completion_path != (campaign / final_api.COMPLETION_NAME).absolute()
            or material_path != (campaign / "materializer_output.json").absolute()):
        raise ValueError("Burned v12 permanent campaign entries differ")

    # Authenticate the source-bound post-claim material without invoking its
    # producer.  Scene values are returned but are not accessed until this
    # entire function has authenticated the candidate and consumed outer.
    _, official_materializer_sources = final_api._official_materializer_binding()
    if (anchor.get("bindings", {}).get(
            "final_materializer_source_closure_sha256")
            != final_api.OFFICIAL_FINAL_MATERIALIZER_SOURCE_CLOSURE_SHA256
            or material.get("producer_sources_sha256")
                != final_api.OFFICIAL_FINAL_MATERIALIZER_SOURCE_CLOSURE_SHA256
            or material.get("producer_sources") != official_materializer_sources
            or completion.get("producer_sources_sha256")
                != anchor.get("bindings", {}).get(
                    "final_controller_source_closure_sha256")):
        raise ValueError("Burned v12 controller/materializer closure differs")

    lock_path, lock, bindings = outer_api._candidate_lock(
        candidate_lock_path, expected_sha256=expected_candidate_lock_sha256)
    inputs = anchor.get("attempt_key_inputs")
    if (not isinstance(inputs, Mapping)
            or inputs.get("candidate_lock_sha256") != file_hash(lock_path)
            or inputs.get("candidate_lock_content_sha256")
                != lock["content_sha256"]
            or inputs.get("actor_sha256") != bindings["actor_sha256"]
            or inputs.get("protocol_sha256") != bindings["protocol_sha256"]
            or inputs.get("runtime_manifest_sha256")
                != bindings["runtime_manifest_sha256"]
            or inputs.get("program_sha256") != bindings["program_sha256"]
            or inputs.get("candidate_source_closure_sha256")
                != bindings["source_closure_sha256"]):
        raise ValueError("Burned v12 final claim and candidate lock differ")
    actor = _regular(actor_path, "frozen Actor", maximum=MAX_NPZ_BYTES,
                     expected_sha256=bindings["actor_sha256"])
    protocol = _regular(protocol_path, "frozen protocol", maximum=MAX_JSON_BYTES,
                        expected_sha256=bindings["protocol_sha256"])
    manifest = _regular(runtime_manifest_path, "frozen runtime manifest",
                        maximum=MAX_JSON_BYTES,
                        expected_sha256=bindings["runtime_manifest_sha256"])
    program = _regular(program_path, "locked program", maximum=MAX_JSON_BYTES,
                       expected_sha256=bindings["program_sha256"])
    selector = _regular(selector_report_path, "locked selector report",
                        maximum=MAX_JSON_BYTES,
                        expected_sha256=bindings["selector_report_sha256"])
    selector_value = collection_api.authenticate_locked_candidate_selector(
        lock=lock, selector_report_path=selector,
        expected_selector_report_sha256=bindings["selector_report_sha256"],
        expected_program_sha256=bindings["program_sha256"])
    program_payload = outer_api._program_payload(program)
    outer_api._authenticate_program_actor_features(
        program_payload=program_payload, actor_path=actor, bindings=bindings)

    result = outer_api.read_saved_result(
        v12_outer_result_path,
        expected_result_sha256=expected_v12_outer_result_sha256,
        permanent_registry=permanent_v12_outer_registry)
    final_api._strict_outer_pass(
        result, bindings=bindings, candidate_lock_sha256=file_hash(lock_path))
    if (inputs.get("outer_attempt_key") != result.get("attempt_key")
            or inputs.get("outer_result_sha256")
                != expected_v12_outer_result_sha256
            or anchor.get("bindings", {}).get("outer_result_content_sha256")
                != result.get("content_sha256")):
        raise ValueError("Burned v12 claim and passed outer result differ")

    registry_path = _regular(
        v12_outer_registry_path, "consumed v12 outer registry",
        maximum=MAX_JSON_BYTES,
        expected_sha256=bindings["fresh_outer_registry_sha256"])
    registry_report_path = _regular(
        v12_outer_registry_report_path, "consumed v12 outer registry report",
        maximum=MAX_JSON_BYTES,
        expected_sha256=bindings["fresh_outer_registry_report_sha256"])
    registry, registry_report, selected_identity_sha = outer_api._registry_bundle(
        registry_path, registry_report_path, bindings=bindings)
    projection, projection_receipt = projection_api.read_saved_projection(
        projection_path=v12_outer_projection_path,
        receipt_path=v12_outer_projection_receipt_path,
        expected_projection_sha256=bindings["outer_hash_projection_sha256"],
        expected_receipt_sha256=bindings["outer_hash_projection_receipt_sha256"])
    if (projection.get("identity", {}).get("selected_identity_sha256")
            != selected_identity_sha):
        raise ValueError("Consumed v12 outer projection identity differs")

    collection_report_path, collection_report = _strict_json_object(
        v12_outer_collection_report_path, "consumed v12 collection report",
        result["outer_collection_report_sha256"])
    collection_bindings = collection_report.get("bindings")
    artifacts = collection_report.get("artifacts")
    if (collection_report.get("version") != collection_api.VERSION
            or collection_report.get("status") != collection_api.STATUS
            or not isinstance(collection_bindings, Mapping)
            or collection_bindings.get("candidate_lock_sha256")
                != file_hash(lock_path)
            or collection_bindings.get("program_sha256") != file_hash(program)
            or collection_bindings.get("rows_sha256") != result["outer_rows_sha256"]
            or not isinstance(artifacts, Mapping)
            or artifacts.get(collection_api.ROWS_NAME, {}).get("file_sha256")
                != result["outer_rows_sha256"]):
        raise ValueError("Consumed v12 outer collection chain differs")
    v12_rows_path = _regular(
        v12_outer_collection_rows_path, "consumed v12 outer rows",
        maximum=MAX_NPZ_BYTES, expected_sha256=result["outer_rows_sha256"])
    v12_rows = collection_api.load_authenticated_rows(
        v12_rows_path, expected_rows_sha256=result["outer_rows_sha256"])
    v12_semantic = _arrays_digest(v12_rows)
    if (v12_semantic != artifacts[collection_api.ROWS_NAME]["semantic_sha256"]
            or v12_semantic != collection_bindings.get("rows_semantic_sha256")):
        raise ValueError("Consumed v12 outer row semantic digest differs")
    ordered_v12 = _decode_hashes(
        v12_rows["observation_hashes"], "consumed v12 outer observation hashes")
    projection_payload = projection.get("projection")
    if (not isinstance(projection_payload, Mapping)
            or projection_payload.get("ordered_observation_hashes") != ordered_v12
            or projection_payload.get("unique_observation_hashes")
                != sorted(set(ordered_v12))):
        raise ValueError("Consumed v12 outer row/projection replay differs")

    v11_closeout = v11_closeout_api.read_saved_closeout(
        v11_closeout_path,
        expected_closeout_sha256=expected_v11_closeout_sha256,
        permanent_attempt_registry=permanent_v11_outer_registry)
    v11_info = v11_closeout.get("consumed_outer")
    if not isinstance(v11_info, Mapping):
        raise ValueError("Consumed v11 closeout rows are missing")
    v11_rows_file = _regular(
        v11_rows_path, "consumed v11 promoted rows", maximum=MAX_NPZ_BYTES,
        expected_sha256=v11_info["rows_sha256"])
    v11_rows = collection_api.load_authenticated_rows(
        v11_rows_file, expected_rows_sha256=v11_info["rows_sha256"])
    if _arrays_digest(v11_rows) != v11_info["rows_semantic_sha256"]:
        raise ValueError("Consumed v11 promoted row semantic digest differs")

    # This is the final authentication step.  Only now validate and return the
    # 64 materialised scenes for development recollection.
    scenes = final_api._validate_material(
        material, anchor=anchor,
        materializer_sources=official_materializer_sources)
    return {
        "anchor": anchor, "completion": completion, "material": material,
        "scenes": scenes, "lock": lock, "bindings": bindings,
        "selector": selector_value, "program_payload": program_payload,
        "actor_path": actor, "protocol_path": protocol,
        "manifest_path": manifest, "program_path": program,
        "v11_closeout": v11_closeout, "v11_rows": v11_rows,
        "v11_rows_path": v11_rows_file,
        "v12_registry": registry, "v12_registry_report": registry_report,
        "v12_projection": projection,
        "v12_projection_receipt": projection_receipt,
        "v12_rows": v12_rows, "v12_rows_path": v12_rows_path,
        "v12_collection_report": collection_report,
        "v12_collection_report_path": collection_report_path,
        "v12_result": result,
        "input_paths": {
            "final_anchor": anchor_path, "final_completion": completion_path,
            "materializer_output": material_path, "candidate_lock": lock_path,
            "selector_report": selector, "program": program,
        },
    }


def _failure_audit(
    *, arrays: Mapping[str, np.ndarray], replay: Mapping[str, np.ndarray],
    program_payload: Mapping[str, Any], environment_steps: int,
    replay_environment_steps: int,
) -> dict[str, Any]:
    collection_api._validate_static_rows(arrays)
    collection_api._validate_static_rows(replay)
    audit_api._assert_exact_replay(arrays, replay)
    if (environment_steps <= 0 or replay_environment_steps != environment_steps
            or not np.all(arrays["split_validation"])):
        raise ValueError("Burned-final deterministic replay accounting differs")
    program = R41DiagnosticPublicTreeProgramV9.from_dict(program_payload)
    probabilities = audit_api._predict(program, arrays["observations"])
    mask = np.ones(len(arrays["observations"]), dtype=np.bool_)
    pairs = rows_api._effective_pairs(arrays, mask)
    pair_bits = metrics_api._pair_group_bits(arrays, pairs)
    metrics = metrics_api._metrics_from_probabilities(
        probabilities, arrays, mask, pairs=pairs, pair_group_bits=pair_bits)
    gate = metrics_api._gate(metrics)
    if len(gate.get("checks", {})) != 9:
        raise ValueError("Burned-final explanation gate registry differs")
    physical = _decode_text(arrays["physical_hashes"], "physical hashes")
    kinds = _decode_text(arrays["kinds"], "row kinds")
    source_states = _decode_hashes(
        arrays["source_state_hashes"], "source state hashes")
    physical_valid = all(
        (kind != "intervention" and value == "")
        or (kind == "intervention" and _HEX.fullmatch(value) is not None)
        for kind, value in zip(kinds, physical)
    )
    result: dict[str, Any] = {
        "status": "passed_nine_gates" if gate["passed"] else "failed_nine_gates",
        "metrics": metrics,
        "nine_gate": gate,
        "failed_checks": sorted(
            name for name, passed in gate["checks"].items() if not passed),
        "coverage": {
            "scenes": len(set(_decode_hashes(
                arrays["scene_fingerprints"], "scene fingerprints"))),
            "rows": len(arrays["observations"]),
            "effective_intervention_pairs": len(pairs),
            "environment_steps": environment_steps,
        },
        "physical_counterfactual_audit": {
            "independent_replay_exact": True,
            "rows_semantic_sha256": _arrays_digest(arrays),
            "replay_rows_semantic_sha256": _arrays_digest(replay),
            "environment_steps_equal": replay_environment_steps == environment_steps,
            "intervention_physical_hashes_valid": physical_valid,
            "source_state_hashes_valid": len(source_states) == len(arrays["observations"]),
            "passed": physical_valid,
        },
        "action_authority": {
            "all_submitted_actions_equal_policy_actions": bool(
                np.all(arrays["submitted_equal"])),
            "runtime_action_overrides": 0,
            "program_controls_runtime_actions": False,
        },
        "post_closeout_development_only": True,
        "formal_ready": False,
    }
    result["content_sha256"] = digest(result)
    return result


def _decode_text(array: np.ndarray, label: str) -> list[str]:
    if array.ndim != 1 or array.dtype.kind != "S":
        raise ValueError(label + " must be one-dimensional byte strings")
    try:
        return list(map(str, np.char.decode(array, "ascii")))
    except UnicodeDecodeError as error:
        raise ValueError(label + " must be ASCII") from error


def _prepare_closeout(**kwargs: Any) -> tuple[
        dict[str, Any], dict[str, bytes], dict[str, np.ndarray]]:
    authenticated = _authenticate_burned_inputs(**kwargs)
    scenes = authenticated["scenes"]
    identities = _public_identity_registry(scenes)
    if identities["family_counts"] != FAMILY_COUNTS:
        raise ValueError("Burned-final family accounting differs")

    burned_rows, environment_steps = final_api._collect_final_rows(
        actor_path=authenticated["actor_path"],
        protocol_path=authenticated["protocol_path"],
        manifest_path=authenticated["manifest_path"], scenes=scenes)
    replay_rows, replay_steps = final_api._collect_final_rows(
        actor_path=authenticated["actor_path"],
        protocol_path=authenticated["protocol_path"],
        manifest_path=authenticated["manifest_path"], scenes=scenes)
    projection_api._validate_projection_replay_arrays(
        burned_rows,
        actor=projection_api.NumPyNativeActor(authenticated["actor_path"]),
        scenes=scenes)
    audit_api._assert_exact_replay(burned_rows, replay_rows)
    failure_audit = _failure_audit(
        arrays=burned_rows, replay=replay_rows,
        program_payload=authenticated["program_payload"],
        environment_steps=environment_steps,
        replay_environment_steps=replay_steps)
    if (not np.all(burned_rows["submitted_equal"])
            or failure_audit["action_authority"]["runtime_action_overrides"] != 0):
        raise ValueError("Burned-final recollection violates Actor authority")

    burned_projection = _observation_projection(
        burned_rows, version=VERSION + ".burned-final-projection.v1",
        source="burned_v12_final_post_closeout_recollection")
    v12_rows = authenticated["v12_rows"]
    v11_rows = authenticated["v11_rows"]
    combined_rows = _combine_rows(v11_rows, v12_rows, burned_rows)
    collection_api._validate_static_rows(combined_rows)
    combined_projection = _observation_projection(
        combined_rows, version=VERSION + ".combined-promoted-projection.v1",
        source="v11_then_v12_outer_then_burned_v12_final")

    v11_info = authenticated["v11_closeout"]["consumed_outer"]
    v11_projection = v11_info["observation_hash_projection"]
    v12_projection_payload = authenticated["v12_projection"]["projection"]
    v12_registry = authenticated["v12_registry"]
    v12_identities = [
        {name: row[name] for name in (
            "batch_index", "family_id", "seed", "fingerprint")}
        for row in v12_registry["selected_outer_identities"]
    ]
    v12_projection_public: dict[str, Any] = {
        "source_projection_sha256": file_hash(Path(
            kwargs["v12_outer_projection_path"]).expanduser().absolute()),
        "source_projection_content_sha256": authenticated[
            "v12_projection"]["content_sha256"],
        "unique_outer_observation_hash_count": len(
            v12_projection_payload["unique_observation_hashes"]),
        "outer_observation_hashes": v12_projection_payload[
            "unique_observation_hashes"],
        "outer_observation_hashes_sha256": digest(
            v12_projection_payload["unique_observation_hashes"]),
        "raw_observations_included": False,
        "actions_included": False,
        "probabilities_included": False,
        "labels_included": False,
    }
    v12_projection_public["content_sha256"] = digest(v12_projection_public)

    sources = producer_sources()
    closeout_key = digest({
        "version": VERSION,
        "attempt_key": authenticated["anchor"]["attempt_key"],
        "final_anchor_sha256": file_hash(authenticated["input_paths"]["final_anchor"]),
        "final_completion_sha256": file_hash(
            authenticated["input_paths"]["final_completion"]),
        "materializer_output_sha256": file_hash(
            authenticated["input_paths"]["materializer_output"]),
        "producer_sources_sha256": digest(sources),
    })
    receipt: dict[str, Any] = {
        "version": VERSION, "status": STATUS, "closeout_key": closeout_key,
        "attempt_key": authenticated["anchor"]["attempt_key"],
        "contract": contract(),
        "burned_final": {
            "attempt_key": authenticated["anchor"]["attempt_key"],
            "claim_consumed": True, "retry_allowed": False,
            "identities": identities["identities"],
            "identities_sha256": identities["identities_sha256"],
            "scene_count": FINAL_SCENE_COUNT,
            "family_counts": identities["family_counts"],
            "rows_sha256": None,
            "rows_semantic_sha256": _arrays_digest(burned_rows),
            "row_count": len(burned_rows["observations"]),
            "post_closeout_deterministic_development_recollection": True,
            "observation_hash_projection": deepcopy(burned_projection),
            "protected_salt_reused": False,
        },
        "consumed_v12_outer": {
            "attempt_key": authenticated["v12_result"]["attempt_key"],
            "identities": v12_identities,
            "identities_sha256": digest(v12_identities),
            "scene_count": len(v12_identities),
            "rows_sha256": file_hash(authenticated["v12_rows_path"]),
            "rows_semantic_sha256": _arrays_digest(v12_rows),
            "row_count": len(v12_rows["observations"]),
            "observation_hash_projection": v12_projection_public,
            "eligible_for_development_use": True,
            "eligible_for_outer_or_final_reuse": False,
        },
        "consumed_v11_outer": {
            "closeout_sha256": kwargs["expected_v11_closeout_sha256"],
            "identities": deepcopy(v11_info["identities"]),
            "identities_sha256": v11_info["identities_sha256"],
            "scene_count": v11_info["scene_count"],
            "rows_sha256": v11_info["rows_sha256"],
            "rows_semantic_sha256": v11_info["rows_semantic_sha256"],
            "row_count": v11_info["row_count"],
            "observation_hash_projection": deepcopy(v11_projection),
        },
        "combined_promoted_development": {
            "source_order": contract()["combined_promoted_source_order"],
            "rows_sha256": None,
            "rows_semantic_sha256": _arrays_digest(combined_rows),
            "row_count": len(combined_rows["observations"]),
            "all_rows_split_validation_true": bool(
                np.all(combined_rows["split_validation"])),
            "observation_hash_projection": deepcopy(combined_projection),
        },
        "failure_audit": failure_audit,
        "bindings": {
            "final_anchor_sha256": file_hash(
                authenticated["input_paths"]["final_anchor"]),
            "final_anchor_content_sha256": authenticated["anchor"]["content_sha256"],
            "final_completion_sha256": file_hash(
                authenticated["input_paths"]["final_completion"]),
            "final_completion_content_sha256": authenticated[
                "completion"]["content_sha256"],
            "materializer_output_sha256": file_hash(
                authenticated["input_paths"]["materializer_output"]),
            "materializer_output_content_sha256": authenticated[
                "material"]["content_sha256"],
            "candidate_lock_sha256": file_hash(
                authenticated["input_paths"]["candidate_lock"]),
            "selector_report_sha256": file_hash(
                authenticated["input_paths"]["selector_report"]),
            "program_sha256": file_hash(authenticated["input_paths"]["program"]),
            "actor_sha256": authenticated["bindings"]["actor_sha256"],
            "protocol_sha256": authenticated["bindings"]["protocol_sha256"],
            "runtime_manifest_sha256": authenticated[
                "bindings"]["runtime_manifest_sha256"],
            "v12_outer_result_sha256": kwargs[
                "expected_v12_outer_result_sha256"],
            "v12_outer_result_content_sha256": authenticated[
                "v12_result"]["content_sha256"],
            "v12_outer_rows_sha256": file_hash(authenticated["v12_rows_path"]),
            "v12_outer_rows_semantic_sha256": _arrays_digest(v12_rows),
            "v11_closeout_sha256": kwargs["expected_v11_closeout_sha256"],
            "v11_rows_sha256": file_hash(authenticated["v11_rows_path"]),
            "v11_rows_semantic_sha256": _arrays_digest(v11_rows),
            "burned_identity_registry_sha256": None,
            "burned_identity_registry_content_sha256": identities[
                "content_sha256"],
            "burned_projection_sha256": None,
            "burned_projection_content_sha256": burned_projection[
                "content_sha256"],
            "combined_projection_sha256": None,
            "combined_projection_content_sha256": combined_projection[
                "content_sha256"],
            "promoted_rows_sha256": None,
            "promoted_rows_semantic_sha256": _arrays_digest(burned_rows),
            "combined_promoted_rows_sha256": None,
            "combined_promoted_rows_semantic_sha256": _arrays_digest(combined_rows),
            "producer_sources_sha256": digest(sources),
        },
        "disposition": {
            "same_protected_final_attempt_retry_permitted": False,
            "burned_final_eligible_for_development_use": True,
            "consumed_v12_outer_eligible_for_development_use": True,
            "burned_final_identity_reuse_permitted": False,
            "burned_final_observation_hash_reuse_permitted": False,
            "consumed_v12_outer_identity_reuse_permitted": False,
            "consumed_v12_outer_observation_hash_reuse_permitted": False,
            "replacement_requires_new_candidate_outer_and_final_claim": True,
            "formal_ready": False,
        },
        "information_boundary": {
            "burned_completion_authenticated_before_scene_access": True,
            "claim_local_materializer_output_authenticated": True,
            "materializer_reinvoked": False,
            "protected_salt_statted_or_read": False,
            "protected_salt_reused": False,
            "scene_snapshots_or_rng_published": False,
            "one_way_observation_hashes_published": True,
            "post_closeout_rows_are_development_only": True,
            "new_outer_or_final_attempt_run": False,
            "runtime_action_override": False,
            "formal_ready": False,
        },
        "producer_sources": sources,
        "producer_sources_sha256": digest(sources),
        "formal_ready": False,
    }
    companions = {
        IDENTITY_NAME: _json_bytes(identities),
        PROMOTED_PROJECTION_NAME: _json_bytes(burned_projection),
        COMBINED_PROJECTION_NAME: _json_bytes(combined_projection),
    }
    return receipt, companions, {
        PROMOTED_ROWS_NAME: burned_rows,
        COMBINED_ROWS_NAME: combined_rows,
    }


def create_closeout(**kwargs: Any) -> dict[str, Any]:
    receipt, _, _ = _prepare_closeout(**kwargs)
    return deepcopy(receipt)


def _materialize_directory(
    directory: Path, receipt: dict[str, Any], companions: Mapping[str, bytes],
    row_artifacts: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    for name, raw in companions.items():
        _write_exclusive(directory / name, raw)
    for name, arrays in row_artifacts.items():
        _write_npz_exclusive(directory / name, arrays)
    receipt = deepcopy(receipt)
    bindings = receipt["bindings"]
    bindings["burned_identity_registry_sha256"] = file_hash(directory / IDENTITY_NAME)
    bindings["burned_projection_sha256"] = file_hash(
        directory / PROMOTED_PROJECTION_NAME)
    bindings["combined_projection_sha256"] = file_hash(
        directory / COMBINED_PROJECTION_NAME)
    bindings["promoted_rows_sha256"] = file_hash(directory / PROMOTED_ROWS_NAME)
    bindings["combined_promoted_rows_sha256"] = file_hash(
        directory / COMBINED_ROWS_NAME)
    receipt["burned_final"]["rows_sha256"] = bindings["promoted_rows_sha256"]
    receipt["combined_promoted_development"]["rows_sha256"] = bindings[
        "combined_promoted_rows_sha256"]
    receipt["content_sha256"] = digest(receipt)
    receipt_raw = _json_bytes(receipt)
    _write_exclusive(directory / RECEIPT_NAME, receipt_raw)
    return receipt, {**companions, RECEIPT_NAME: receipt_raw}


def build(*, output: str | Path, permanent_closeout_registry: str | Path,
          **kwargs: Any) -> dict[str, Any]:
    receipt, companions, row_artifacts = _prepare_closeout(**kwargs)
    destination = Path(output).expanduser().absolute()
    parent = _directory(destination.parent, "v13 closeout output parent")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("v13 closeout output already exists")
    permanent = _directory(permanent_closeout_registry,
                           "permanent v13 closeout registry")
    campaign = permanent / receipt["closeout_key"]
    if campaign.exists() or campaign.is_symlink():
        raise FileExistsError("v13 permanent closeout campaign already exists")

    temporary = Path(tempfile.mkdtemp(
        prefix="." + destination.name + ".tmp-", dir=parent)).absolute()
    permanent_temporary = Path(tempfile.mkdtemp(
        prefix="." + receipt["closeout_key"] + ".tmp-", dir=permanent)).absolute()
    try:
        receipt, _ = _materialize_directory(
            temporary, receipt, companions, row_artifacts)
        # Write the exact already-frozen bytes into the permanent append-only
        # campaign.  Re-serialising NPZ files could change ZIP metadata even
        # when every array is identical, so byte copying is the stricter bind.
        for name in (
            RECEIPT_NAME, IDENTITY_NAME, PROMOTED_PROJECTION_NAME,
            COMBINED_PROJECTION_NAME, PROMOTED_ROWS_NAME, COMBINED_ROWS_NAME,
        ):
            _write_exclusive(permanent_temporary / name,
                             (temporary / name).read_bytes())
            if ((temporary / name).read_bytes()
                    != (permanent_temporary / name).read_bytes()):
                raise RuntimeError("Permanent closeout artifact bytes differ: " + name)
        os.rename(permanent_temporary, campaign)
        permanent_temporary = None
        os.rename(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        if permanent_temporary is not None:
            shutil.rmtree(permanent_temporary, ignore_errors=True)
    return deepcopy(receipt)


def read_saved_closeout(
    path: str | Path, *, expected_closeout_sha256: str,
    permanent_closeout_registry: str | Path,
) -> dict[str, Any]:
    receipt_path, receipt_raw, value = _strict_json(
        path, "v13 burned-final closeout",
        expected_sha256=expected_closeout_sha256)
    if (value.get("version") != VERSION or value.get("status") != STATUS
            or value.get("contract") != contract() or not _content_valid(value)
            or value.get("formal_ready") is not False
            or value.get("burned_final", {}).get("retry_allowed") is not False
            or value.get("burned_final", {}).get("protected_salt_reused") is not False
            or value.get("disposition", {}).get(
                "same_protected_final_attempt_retry_permitted") is not False
            or value.get("information_boundary", {}).get(
                "protected_salt_statted_or_read") is not False
            or value.get("producer_sources_sha256")
                != digest(value.get("producer_sources", {}))):
        raise ValueError("Saved v13 burned-final closeout semantics differ")
    directory = _directory(receipt_path.parent, "v13 closeout artifact directory")
    expected_names = {
        RECEIPT_NAME, IDENTITY_NAME, PROMOTED_PROJECTION_NAME,
        COMBINED_PROJECTION_NAME, PROMOTED_ROWS_NAME, COMBINED_ROWS_NAME,
    }
    if {entry.name for entry in directory.iterdir()} != expected_names:
        raise ValueError("Saved v13 closeout artifact set differs")
    bindings = value.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("Saved v13 closeout bindings differ")
    json_bindings = {
        IDENTITY_NAME: ("burned_identity_registry_sha256",
                        "burned_identity_registry_content_sha256"),
        PROMOTED_PROJECTION_NAME: (
            "burned_projection_sha256", "burned_projection_content_sha256"),
        COMBINED_PROJECTION_NAME: (
            "combined_projection_sha256", "combined_projection_content_sha256"),
    }
    for name, (file_key, content_key) in json_bindings.items():
        _, _, payload = _strict_json(
            directory / name, name, expected_sha256=bindings[file_key])
        if not _content_valid(payload) or payload["content_sha256"] != bindings[content_key]:
            raise ValueError("Saved v13 closeout JSON companion differs: " + name)
    for name, file_key, semantic_key in (
        (PROMOTED_ROWS_NAME, "promoted_rows_sha256",
         "promoted_rows_semantic_sha256"),
        (COMBINED_ROWS_NAME, "combined_promoted_rows_sha256",
         "combined_promoted_rows_semantic_sha256"),
    ):
        arrays = collection_api.load_authenticated_rows(
            directory / name, expected_rows_sha256=bindings[file_key])
        if _arrays_digest(arrays) != bindings[semantic_key]:
            raise ValueError("Saved v13 closeout row companion differs: " + name)
    permanent = _directory(permanent_closeout_registry,
                           "permanent v13 closeout registry")
    campaign = _directory(permanent / value["closeout_key"],
                          "permanent v13 closeout campaign")
    if {entry.name for entry in campaign.iterdir()} != expected_names:
        raise ValueError("Permanent v13 closeout artifact set differs")
    for name in expected_names:
        if (campaign / name).read_bytes() != (directory / name).read_bytes():
            raise ValueError("Permanent v13 closeout artifact differs: " + name)
    if (receipt_raw != (campaign / RECEIPT_NAME).read_bytes()
            or file_hash(receipt_path) != expected_closeout_sha256):
        raise ValueError("Permanent v13 closeout receipt differs")
    return deepcopy(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "final-anchor", "expected-final-anchor-sha256", "final-completion",
        "expected-final-completion-sha256", "materializer-output",
        "expected-materializer-output-sha256", "permanent-final-registry",
        "candidate-lock", "expected-candidate-lock-sha256", "actor", "protocol",
        "runtime-manifest", "program", "selector-report", "v12-outer-registry",
        "v12-outer-registry-report", "v12-outer-projection",
        "v12-outer-projection-receipt", "v12-outer-collection-rows",
        "v12-outer-collection-report", "v12-outer-result",
        "expected-v12-outer-result-sha256", "permanent-v12-outer-registry",
        "v11-closeout", "expected-v11-closeout-sha256",
        "permanent-v11-outer-registry", "v11-rows",
        "permanent-closeout-registry", "output",
    ):
        parser.add_argument("--" + name, required=True)
    args = vars(parser.parse_args(argv))
    result = build(
        output=args.pop("output"),
        permanent_closeout_registry=args.pop("permanent_closeout_registry"),
        **{name + "_path" if name in {
            "final_anchor", "final_completion", "materializer_output",
            "candidate_lock", "actor", "protocol", "runtime_manifest",
            "program", "selector_report", "v12_outer_registry",
            "v12_outer_registry_report", "v12_outer_projection",
            "v12_outer_projection_receipt", "v12_outer_collection_rows",
            "v12_outer_collection_report", "v12_outer_result",
            "v11_closeout", "v11_rows",
        } else name: value for name, value in args.items()},
    )
    print(canonical({
        "status": result["status"], "closeout_key": result["closeout_key"],
        "burned_final_failed_checks": result["failure_audit"]["failed_checks"],
        "combined_promoted_rows": result[
            "combined_promoted_development"]["row_count"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VERSION", "STATUS", "RECEIPT_NAME", "IDENTITY_NAME",
    "PROMOTED_PROJECTION_NAME", "COMBINED_PROJECTION_NAME",
    "PROMOTED_ROWS_NAME", "COMBINED_ROWS_NAME", "contract",
    "producer_sources", "create_closeout", "build", "read_saved_closeout",
    "main",
]
