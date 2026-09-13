"""Close the consumed v11 outer attempt and promote its rows to development.

The v11 one-shot scorer irrevocably claimed and scored its fresh outer, then
failed one explanation gate.  The 64 scenes and their labels are therefore
development-exposed and can never be reused as an outer set.  This module
authenticates the exact frozen campaign, publishes its public identities and
one-way observation hashes, and binds the unopened row archive by both file
and producer-recorded semantic SHA-256 so a later selector can explicitly
promote it to development data.

No NPZ member, protected-final artifact, or private salt is opened here.
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

from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import canonical, digest, file_hash


VERSION = "warehouse-r41-diagnostic-outer-attempt-closeout.v12"
STATUS = "consumed_v11_outer_irrevocably_closed"
RECEIPT_NAME = "closeout_receipt.json"
SCENE_COUNT = 64
ROW_COUNT = 36_317
UNIQUE_OBSERVATION_COUNT = 31_522
FAMILY_COUNTS = {
    "conflict_family_01": 11,
    "conflict_family_02": 11,
    "conflict_family_03": 11,
    "conflict_family_04": 11,
    "conflict_family_05": 10,
    "conflict_family_06": 10,
}

REGISTRY_VERSION = "warehouse-r41-diagnostic-rcpd-v11-fresh-outer-registry.v1"
REGISTRY_STATUS = "frozen_hash_screened_pending_outer_collection"
PROJECTION_VERSION = "warehouse-r41-diagnostic-outer-hash-projection.v11"
PROJECTION_STATUS = "frozen_label_blind_ordered_outer_hash_projection"
COLLECTION_VERSION = "warehouse-r41-diagnostic-outer-collection.v11"
COLLECTION_STATUS = "collected_unscored"
SELECTOR_VERSION = "warehouse-r41-diagnostic-rcpd-v11-fit-selector.v1"
SELECTOR_STATUS = "locked_development_candidate_pending_fresh_outer"
LOCK_SCHEMA = "warehouse_r41_diagnostic_rcpd_v11_candidate_lock_v1"
PROGRAM_VERSION = "warehouse-r41-diagnostic-public-tree-program.v9"
GRID_VERSION = "warehouse-r41-diagnostic-rcpd-v9-candidate-grid.v1"
OUTER_VERSION = "warehouse-r41-diagnostic-rcpd-v11-outer-once.v1"
ATTEMPT_KEY = "aec4faad97b3f7ced0e399f392eae9e37a61df6b89a899dc4e5d482c65589d2b"

# Exact bytes of the already-consumed v11 campaign.
EXPECTED_REGISTRY_SHA256 = "870ac69378cb76440ddea8518b9575b55c520ce4d5cc6eb87e059bd7df83dbf2"
EXPECTED_REGISTRY_REPORT_SHA256 = "1ae827f762eaa4ad28e958b591a549c0dc91b554d192620f4d297373d0cbf9c5"
EXPECTED_PROJECTION_SHA256 = "b3b78faf6ce83c7528b05dca93acf7787314748b81e28b501ebbd8bd9b74b3d5"
EXPECTED_PROJECTION_RECEIPT_SHA256 = "dcb5772154a9c3879f25b8732de7b2d7d1baf45eb39b56146bcbe79bab22a83b"
EXPECTED_COLLECTION_ROWS_SHA256 = "d8f84d43c3fe01445c664353fb19b79bde196066c6679ff77e24cd65249b7d92"
EXPECTED_COLLECTION_ROWS_SEMANTIC_SHA256 = "ba5b219e8dc1f959d6865d9a7e9063d42fdff815297dcd3b884584b842d6559c"
EXPECTED_COLLECTION_REPORT_SHA256 = "738e049aed3d752b3cc8b75f962f2b0ebd9fd0e8691051bf17a91eed71c2799e"
EXPECTED_CANDIDATE_LOCK_SHA256 = "e078ed4a8ca025f8f672889d590566d5daf35810d80987fb7bdbd9de6837c144"
EXPECTED_SELECTOR_REPORT_SHA256 = "3e12d2b526c3868d9dc4d9ccea6a3a8bf2a3e98e354278ccd229b22ab1edd9a3"
EXPECTED_PROGRAM_SHA256 = "f2d4c3d39e7fa519c5e0051757c5efbe733a21f5896af7538e137db67c5f3da5"
EXPECTED_CANDIDATE_GRID_SHA256 = "900578faaf4aadc4d4b0d25494d2468f654d644a3aa3d23f9b1b155dc149680e"
EXPECTED_ATTEMPT_ANCHOR_SHA256 = "f2657728674f20dd5242488c86078ca4593a699f116e0b092480705370d2dd6b"
EXPECTED_ATTEMPT_RESULT_SHA256 = "54d63f18297afae928abcf1d4f3357c59554d8a0ccbb5a3318b7d482456b40a2"

MAX_JSON_BYTES = 512 * 1024 * 1024
MAX_OPAQUE_BYTES = 2 * 1024 * 1024 * 1024
_HEX = re.compile(r"[0-9a-f]{64}\Z")


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


def _sources_valid(value: Any, claimed: Any) -> bool:
    return (isinstance(value, Mapping) and bool(value)
            and all(type(name) is str and bool(name)
                    and type(checksum) is str
                    and _HEX.fullmatch(checksum) is not None
                    for name, checksum in value.items())
            and claimed == digest(dict(sorted(value.items()))))


def _directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    if not path.is_dir() or path.is_symlink() or path.resolve() != path:
        raise ValueError(label + " must be a canonical directory")
    return path


def _regular(value: str | Path, label: str, *, maximum: int) -> Path:
    path = Path(value).expanduser().absolute()
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or path.stat(follow_symlinks=False).st_size <= 0
            or path.stat(follow_symlinks=False).st_size > maximum):
        raise ValueError(label + " must be a bounded canonical regular file")
    return path


def _strict_json(value: str | Path, label: str, *, expected_sha256: str
                 ) -> tuple[Path, bytes, dict[str, Any]]:
    path = _regular(value, label, maximum=MAX_JSON_BYTES)
    expected = _sha(expected_sha256, label + " SHA-256")
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError("Exact " + label + " bytes required")

    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in rows:
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
    if not isinstance(parsed, dict) or file_hash(path) != expected:
        raise RuntimeError(label + " changed during read")
    return path, raw, parsed


def _opaque(value: str | Path, label: str, *, expected_sha256: str) -> Path:
    path = _regular(value, label, maximum=MAX_OPAQUE_BYTES)
    if file_hash(path) != _sha(expected_sha256, label + " SHA-256"):
        raise ValueError("Exact " + label + " bytes required")
    return path


def _identity(row: Any, label: str) -> dict[str, Any]:
    if not isinstance(row, Mapping) or set(row) != {
            "batch_index", "family_id", "seed", "fingerprint"}:
        raise ValueError(label + " identity fields differ")
    value = {key: row[key] for key in (
        "batch_index", "family_id", "seed", "fingerprint")}
    if (type(value["batch_index"]) is not int or value["batch_index"] < 0
            or value["family_id"] not in FAMILY_COUNTS
            or type(value["seed"]) is not int or value["seed"] < 0
            or type(value["fingerprint"]) is not str
            or _HEX.fullmatch(value["fingerprint"]) is None):
        raise ValueError(label + " identity values differ")
    return value


def _registry_identities(registry: Mapping[str, Any],
                         report: Mapping[str, Any], *, registry_sha256: str
                         ) -> list[dict[str, Any]]:
    selected = registry.get("selected_outer_identities")
    scenes = registry.get("development_outer")
    if (registry.get("version") != REGISTRY_VERSION
            or registry.get("status") != REGISTRY_STATUS
            or not _content_valid(registry)
            or registry.get("program_access") is not False
            or registry.get("program_predictions_access") is not False
            or registry.get("action_labels_access") is not False
            or registry.get("probabilities_access") is not False
            or registry.get("final_audit_rows_access") is not False
            or registry.get("formal_ready") is not False
            or not _sources_valid(registry.get("producer_sources"),
                                  registry.get("producer_sources_sha256"))
            or not isinstance(selected, list) or len(selected) != SCENE_COUNT
            or not isinstance(scenes, list) or len(scenes) != SCENE_COUNT):
        raise ValueError("Consumed v11 registry semantics differ")
    identities: list[dict[str, Any]] = []
    for index, (raw, scene) in enumerate(zip(selected, scenes)):
        identity = _identity(raw, "consumed v11 outer")
        if (not isinstance(scene, Mapping)
                or scene.get("id") != f"diagnostic_v11_fresh_outer_{index:04d}"
                or any(scene.get(name) != identity[name]
                       for name in ("family_id", "seed", "fingerprint"))):
            raise ValueError("Consumed v11 materialised identity differs")
        identities.append(identity)
    if (len({row["seed"] for row in identities}) != SCENE_COUNT
            or len({row["fingerprint"] for row in identities}) != SCENE_COUNT
            or dict(sorted(Counter(row["family_id"] for row in identities).items()))
                != FAMILY_COUNTS):
        raise ValueError("Consumed v11 identity population differs")
    selection = report.get("selection")
    if (report.get("version") != REGISTRY_VERSION
            or report.get("status") != REGISTRY_STATUS
            or not _content_valid(report)
            or report.get("formal_ready") is not False
            or report.get("registry_file_sha256") != registry_sha256
            or report.get("registry_content_sha256")
                != registry["content_sha256"]
            or report.get("bindings") != registry.get("bindings")
            or report.get("statistics") != registry.get("statistics")
            or report.get("information_boundary")
                != registry.get("information_boundary")
            or report.get("producer_sources")
                != registry.get("producer_sources")
            or not isinstance(selection, Mapping)
            or selection.get("selected_identity_sha256") != digest(identities)):
        raise ValueError("Consumed v11 registry report differs")
    return identities


def _validate_chain(
    *, registry: Mapping[str, Any], report: Mapping[str, Any],
    projection: Mapping[str, Any], projection_receipt: Mapping[str, Any],
    collection: Mapping[str, Any], lock: Mapping[str, Any],
    selector: Mapping[str, Any], program: Mapping[str, Any],
    grid: Mapping[str, Any], anchor: Mapping[str, Any], result: Mapping[str, Any],
    identities: Sequence[Mapping[str, Any]],
) -> None:
    identity_sha = digest(list(identities))
    payload = projection.get("projection")
    hashes = payload.get("unique_observation_hashes") \
        if isinstance(payload, Mapping) else None
    if (projection.get("version") != PROJECTION_VERSION
            or projection.get("status") != PROJECTION_STATUS
            or not _content_valid(projection)
            or projection.get("identity", {}).get("registry_file_sha256")
                != EXPECTED_REGISTRY_SHA256
            or projection.get("identity", {}).get("registry_content_sha256")
                != registry["content_sha256"]
            or projection.get("identity", {}).get("selected_identity_sha256")
                != identity_sha
            or not isinstance(payload, Mapping)
            or payload.get("row_count") != ROW_COUNT
            or not isinstance(hashes, list)
            or hashes != sorted(set(hashes))
            or len(hashes) != UNIQUE_OBSERVATION_COUNT
            or any(type(value) is not str or _HEX.fullmatch(value) is None
                   for value in hashes)
            or payload.get("unique_observation_count") != len(hashes)
            or payload.get("unique_observation_hashes_sha256") != digest(hashes)):
        raise ValueError("Consumed v11 projection differs")
    receipt_bindings = projection_receipt.get("bindings")
    if (projection_receipt.get("version") != PROJECTION_VERSION + ".receipt.v1"
            or projection_receipt.get("status") != PROJECTION_STATUS
            or not _content_valid(projection_receipt)
            or projection_receipt.get("program_accessed") is not False
            or projection_receipt.get("protected_final_access") is not False
            or not isinstance(receipt_bindings, Mapping)
            or receipt_bindings.get("fresh_outer_registry_sha256")
                != EXPECTED_REGISTRY_SHA256
            or receipt_bindings.get("fresh_outer_registry_report_sha256")
                != EXPECTED_REGISTRY_REPORT_SHA256
            or receipt_bindings.get("outer_hash_projection_sha256")
                != EXPECTED_PROJECTION_SHA256
            or receipt_bindings.get("outer_hash_projection_content_sha256")
                != projection["content_sha256"]):
        raise ValueError("Consumed v11 projection receipt differs")
    lock_bindings = lock.get("bindings")
    if (lock.get("schema_version") != LOCK_SCHEMA
            or lock.get("status") != "locked" or not _content_valid(lock)
            or lock.get("formal_ready") is not False
            or not isinstance(lock_bindings, Mapping)
            or lock_bindings.get("fresh_outer_registry_sha256")
                != EXPECTED_REGISTRY_SHA256
            or lock_bindings.get("fresh_outer_registry_report_sha256")
                != EXPECTED_REGISTRY_REPORT_SHA256
            or lock_bindings.get("outer_hash_projection_sha256")
                != EXPECTED_PROJECTION_SHA256
            or lock_bindings.get("outer_hash_projection_receipt_sha256")
                != EXPECTED_PROJECTION_RECEIPT_SHA256
            or lock_bindings.get("program_sha256") != EXPECTED_PROGRAM_SHA256
            or lock_bindings.get("selector_report_sha256")
                != EXPECTED_SELECTOR_REPORT_SHA256
            or lock_bindings.get("candidate_grid_sha256")
                != EXPECTED_CANDIDATE_GRID_SHA256
            or lock.get("selection", {}).get("fresh_outer_scored") is not False
            or lock.get("selection", {}).get(
                "two_salt_three_fold_development_cv_passed") is not True):
        raise ValueError("Consumed v11 candidate lock differs")
    if (selector.get("version") != SELECTOR_VERSION
            or selector.get("status") != SELECTOR_STATUS
            or not _content_valid(selector)
            or selector.get("formal_ready") is not False
            or selector.get("bindings", {}).get("program_sha256")
                != EXPECTED_PROGRAM_SHA256
            or selector.get("selection", {}).get("selected_config_sha256")
                != lock.get("selection", {}).get("selected_config_sha256")
            or selector.get("final_fit", {}).get("program_file_sha256")
                != EXPECTED_PROGRAM_SHA256
            or program.get("version") != PROGRAM_VERSION
            or program.get("metadata", {}).get("fit_config_sha256")
                != selector.get("selection", {}).get("selected_config_sha256")
            or program.get("metadata", {}).get("runtime_action_override") is not False
            or grid.get("version") != GRID_VERSION or not _content_valid(grid)):
        raise ValueError("Consumed v11 selector chain differs")
    collection_bindings = collection.get("bindings")
    collection_state = collection.get("collection")
    artifacts = collection.get("artifacts")
    if (collection.get("version") != COLLECTION_VERSION
            or collection.get("status") != COLLECTION_STATUS
            or not _content_valid(collection)
            or collection.get("formal_ready") is not False
            or not isinstance(collection_bindings, Mapping)
            or collection_bindings.get("fresh_outer_registry_sha256")
                != EXPECTED_REGISTRY_SHA256
            or collection_bindings.get("fresh_outer_registry_report_sha256")
                != EXPECTED_REGISTRY_REPORT_SHA256
            or collection_bindings.get("candidate_lock_sha256")
                != EXPECTED_CANDIDATE_LOCK_SHA256
            or collection_bindings.get("selector_report_sha256")
                != EXPECTED_SELECTOR_REPORT_SHA256
            or collection_bindings.get("program_sha256")
                != EXPECTED_PROGRAM_SHA256
            or not isinstance(artifacts, Mapping)
            or artifacts.get("rows.npz", {}).get("file_sha256")
                != EXPECTED_COLLECTION_ROWS_SHA256
            or artifacts.get("rows.npz", {}).get("semantic_sha256")
                != EXPECTED_COLLECTION_ROWS_SEMANTIC_SHA256
            or not isinstance(collection_state, Mapping)
            or collection_state.get("scene_count") != SCENE_COUNT
            or collection_state.get("row_count") != ROW_COUNT
            or collection_state.get("scored") is not False
            or collection_state.get("score_computed") is not False
            or collection_state.get(
                "all_submitted_actions_equal_policy_actions") is not True):
        raise ValueError("Consumed v11 collection differs")
    inputs, anchor_bindings = anchor.get("attempt_key_inputs"), anchor.get("bindings")
    if (anchor.get("version") != OUTER_VERSION + ".attempt-anchor.v1"
            or anchor.get("status") != "fresh_outer_attempt_irrevocably_claimed"
            or anchor.get("attempt_key") != ATTEMPT_KEY
            or not _content_valid(anchor)
            or anchor.get("retry_permitted") is not False
            or anchor.get("private_outer_fields_read_before_claim") is not False
            or anchor.get("protected_final_access") is not False
            or not isinstance(inputs, Mapping) or digest(dict(inputs)) != ATTEMPT_KEY
            or inputs.get("selected_identity_sha256") != identity_sha
            or inputs.get("fresh_outer_registry_sha256")
                != EXPECTED_REGISTRY_SHA256
            or inputs.get("outer_hash_projection_sha256")
                != EXPECTED_PROJECTION_SHA256
            or not isinstance(anchor_bindings, Mapping)
            or anchor_bindings.get("candidate_lock_sha256")
                != EXPECTED_CANDIDATE_LOCK_SHA256
            or anchor_bindings.get("outer_collection_report_sha256")
                != EXPECTED_COLLECTION_REPORT_SHA256
            or anchor_bindings.get("outer_rows_sha256")
                != EXPECTED_COLLECTION_ROWS_SHA256
            or anchor.get("preflight", {}).get("outer_scene_count") != SCENE_COUNT
            or anchor.get("preflight", {}).get("outer_row_count") != ROW_COUNT):
        raise ValueError("Consumed v11 permanent attempt anchor differs")
    checks = result.get("gate", {}).get("checks")
    execution = result.get("execution")
    if (result.get("version") != OUTER_VERSION
            or result.get("status") != "failed_fresh_development_outer_gates"
            or result.get("attempt_key") != ATTEMPT_KEY
            or not _content_valid(result)
            or result.get("attempt_anchor_content_sha256")
                != anchor["content_sha256"]
            or result.get("formal_ready") is not False
            or result.get("explanation_eligible") is not False
            or not isinstance(execution, Mapping)
            or execution.get("outer_consumed") is not True
            or execution.get("retry_permitted") is not False
            or execution.get("protected_final_access") is not False
            or result.get("outer_rows_sha256") != EXPECTED_COLLECTION_ROWS_SHA256
            or result.get("outer_collection_report_sha256")
                != EXPECTED_COLLECTION_REPORT_SHA256
            or not isinstance(checks, Mapping) or result.get("gate", {}).get("passed") is not False
            or sorted(key for key, passed in checks.items() if passed is False)
                != ["effective_intervention_direction_shared_charger"]):
        raise ValueError("Consumed v11 permanent attempt result differs")


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "source": "authenticated failed-gate v11 one-shot outer",
        "consumed_scene_count": SCENE_COUNT,
        "consumed_outer_is_development_exposed": True,
        "promoted_rows_require_exact_file_and_semantic_sha256": True,
        "outer_identity_reuse_permitted": False,
        "row_archive_members_opened": False,
        "one_way_observation_hash_projection_published": True,
        "protected_final_access": False,
        "private_salt_access": False,
        "formal_ready": False,
    }


def create_closeout(
    *, registry_path: str | Path, registry_report_path: str | Path,
    projection_path: str | Path, projection_receipt_path: str | Path,
    collection_rows_path: str | Path, collection_report_path: str | Path,
    collection_projection_copy_path: str | Path,
    candidate_lock_path: str | Path, selector_report_path: str | Path,
    program_path: str | Path, candidate_grid_path: str | Path,
    attempt_anchor_path: str | Path, attempt_result_path: str | Path,
    permanent_attempt_registry: str | Path,
) -> dict[str, Any]:
    sources = producer_sources()
    registry_file, _, registry = _strict_json(
        registry_path, "consumed v11 registry",
        expected_sha256=EXPECTED_REGISTRY_SHA256)
    report_file, _, report = _strict_json(
        registry_report_path, "consumed v11 registry report",
        expected_sha256=EXPECTED_REGISTRY_REPORT_SHA256)
    identities = _registry_identities(
        registry, report, registry_sha256=file_hash(registry_file))
    projection_file, projection_raw, projection = _strict_json(
        projection_path, "consumed v11 projection",
        expected_sha256=EXPECTED_PROJECTION_SHA256)
    projection_receipt_file, _, projection_receipt = _strict_json(
        projection_receipt_path, "consumed v11 projection receipt",
        expected_sha256=EXPECTED_PROJECTION_RECEIPT_SHA256)
    rows_file = _opaque(
        collection_rows_path, "consumed v11 rows",
        expected_sha256=EXPECTED_COLLECTION_ROWS_SHA256)
    collection_file, _, collection = _strict_json(
        collection_report_path, "consumed v11 collection report",
        expected_sha256=EXPECTED_COLLECTION_REPORT_SHA256)
    projection_copy = _opaque(
        collection_projection_copy_path, "consumed v11 projection copy",
        expected_sha256=EXPECTED_PROJECTION_SHA256)
    if projection_copy.read_bytes() != projection_raw:
        raise ValueError("Consumed v11 collection projection copy differs")
    lock_file, _, lock = _strict_json(
        candidate_lock_path, "consumed v11 candidate lock",
        expected_sha256=EXPECTED_CANDIDATE_LOCK_SHA256)
    selector_file, _, selector = _strict_json(
        selector_report_path, "consumed v11 selector report",
        expected_sha256=EXPECTED_SELECTOR_REPORT_SHA256)
    program_file, _, program = _strict_json(
        program_path, "consumed v11 program", expected_sha256=EXPECTED_PROGRAM_SHA256)
    grid_file, _, grid = _strict_json(
        candidate_grid_path, "consumed v11 candidate grid",
        expected_sha256=EXPECTED_CANDIDATE_GRID_SHA256)
    permanent = _directory(permanent_attempt_registry, "permanent v11 attempt registry")
    campaign = _directory(permanent / ATTEMPT_KEY, "permanent consumed v11 campaign")
    if {entry.name for entry in campaign.iterdir()} != {
            "attempt_anchor.json", "outer_result.json"}:
        raise ValueError("Permanent consumed v11 campaign entries differ")
    expected_anchor, expected_result = (
        campaign / "attempt_anchor.json", campaign / "outer_result.json")
    if (Path(attempt_anchor_path).expanduser().absolute().resolve() != expected_anchor
            or Path(attempt_result_path).expanduser().absolute().resolve()
                != expected_result):
        raise ValueError("Attempt files must be the permanent v11 campaign files")
    anchor_file, _, anchor = _strict_json(
        expected_anchor, "permanent consumed v11 anchor",
        expected_sha256=EXPECTED_ATTEMPT_ANCHOR_SHA256)
    result_file, _, result = _strict_json(
        expected_result, "permanent consumed v11 result",
        expected_sha256=EXPECTED_ATTEMPT_RESULT_SHA256)
    _validate_chain(
        registry=registry, report=report, projection=projection,
        projection_receipt=projection_receipt, collection=collection,
        lock=lock, selector=selector, program=program, grid=grid,
        anchor=anchor, result=result, identities=identities)
    projected = projection["projection"]
    bindings = {
        "v11_registry_sha256": file_hash(registry_file),
        "v11_registry_content_sha256": registry["content_sha256"],
        "v11_registry_report_sha256": file_hash(report_file),
        "v11_registry_report_content_sha256": report["content_sha256"],
        "v11_projection_sha256": file_hash(projection_file),
        "v11_projection_content_sha256": projection["content_sha256"],
        "v11_projection_receipt_sha256": file_hash(projection_receipt_file),
        "v11_collection_rows_sha256": file_hash(rows_file),
        "v11_collection_rows_semantic_sha256": (
            EXPECTED_COLLECTION_ROWS_SEMANTIC_SHA256),
        "v11_collection_report_sha256": file_hash(collection_file),
        "v11_collection_report_content_sha256": collection["content_sha256"],
        "v11_collection_projection_copy_sha256": file_hash(projection_copy),
        "v11_candidate_lock_sha256": file_hash(lock_file),
        "v11_candidate_lock_content_sha256": lock["content_sha256"],
        "v11_selector_report_sha256": file_hash(selector_file),
        "v11_selector_report_content_sha256": selector["content_sha256"],
        "v11_program_sha256": file_hash(program_file),
        "v11_candidate_grid_sha256": file_hash(grid_file),
        "v11_attempt_anchor_sha256": file_hash(anchor_file),
        "v11_attempt_anchor_content_sha256": anchor["content_sha256"],
        "v11_attempt_result_sha256": file_hash(result_file),
        "v11_attempt_result_content_sha256": result["content_sha256"],
        "contract_sha256": digest(contract()),
    }
    failed_checks = sorted(
        key for key, passed in result["gate"]["checks"].items()
        if passed is False)
    receipt: dict[str, Any] = {
        "version": VERSION,
        "status": STATUS,
        "attempt_key": ATTEMPT_KEY,
        "contract": contract(),
        "bindings": bindings,
        "consumed_outer": {
            "scene_count": SCENE_COUNT,
            "identities": deepcopy(identities),
            "identities_sha256": digest(identities),
            "family_counts": deepcopy(FAMILY_COUNTS),
            "row_count": ROW_COUNT,
            "rows_sha256": EXPECTED_COLLECTION_ROWS_SHA256,
            "rows_semantic_sha256": EXPECTED_COLLECTION_ROWS_SEMANTIC_SHA256,
            "observation_hash_projection": {
                "source_projection_sha256": EXPECTED_PROJECTION_SHA256,
                "source_projection_content_sha256": projection["content_sha256"],
                "outer_observation_hashes": deepcopy(
                    projected["unique_observation_hashes"]),
                "unique_outer_observation_hash_count": (
                    projected["unique_observation_count"]),
                "outer_observation_hashes_sha256": (
                    projected["unique_observation_hashes_sha256"]),
                "raw_observations_included": False,
                "actions_included": False,
                "probabilities_included": False,
                "labels_included": False,
            },
        },
        "failure": {
            "status": result["status"],
            "outer_consumed": True,
            "retry_permitted": False,
            "metrics_produced": True,
            "failed_checks": failed_checks,
            "gate_passed": False,
        },
        "disposition": {
            "eligible_for_development_use": True,
            "eligible_for_outer_claim": False,
            "outer_identity_reuse_permitted": False,
            "replacement_outer_requires_new_identities": True,
            "formal_ready": False,
        },
        "information_boundary": {
            "rows_authenticated_by_exact_file_hash": True,
            "rows_authenticated_by_producer_semantic_hash": True,
            "row_archive_members_opened": False,
            "raw_observations_read": False,
            "actions_probabilities_weights_or_labels_read": False,
            "program_executed": False,
            "protected_final_access": False,
            "private_salt_access": False,
            "formal_ready": False,
        },
        "producer_sources": sources,
        "producer_sources_sha256": digest(sources),
        "formal_ready": False,
    }
    receipt["content_sha256"] = digest(receipt)
    if producer_sources() != sources:
        raise RuntimeError("v12 closeout source closure changed during build")
    return receipt


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (canonical(value) + "\n").encode("utf-8")


def _write_exclusive(path: Path, raw: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                         | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def build(*, output: str | Path, **kwargs: Any) -> dict[str, Any]:
    receipt = create_closeout(**kwargs)
    destination = Path(output).expanduser().absolute()
    parent = _directory(destination.parent, "v12 closeout output parent")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("v12 closeout output already exists")
    temporary = Path(tempfile.mkdtemp(
        prefix="." + destination.name + ".tmp-", dir=parent)).absolute()
    try:
        _write_exclusive(temporary / RECEIPT_NAME, _json_bytes(receipt))
        os.rename(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
    return deepcopy(receipt)


def read_saved_closeout(
    path: str | Path, *, expected_closeout_sha256: str,
    permanent_attempt_registry: str | Path,
) -> dict[str, Any]:
    receipt_file, _, value = _strict_json(
        path, "v12 consumed-v11 closeout",
        expected_sha256=expected_closeout_sha256)
    consumed = value.get("consumed_outer")
    failure = value.get("failure")
    disposition = value.get("disposition")
    boundary = value.get("information_boundary")
    projection = consumed.get("observation_hash_projection") \
        if isinstance(consumed, Mapping) else None
    hashes = projection.get("outer_observation_hashes") \
        if isinstance(projection, Mapping) else None
    if (value.get("version") != VERSION or value.get("status") != STATUS
            or not _content_valid(value) or value.get("formal_ready") is not False
            or value.get("contract") != contract()
            or not isinstance(consumed, Mapping)
            or consumed.get("scene_count") != SCENE_COUNT
            or consumed.get("row_count") != ROW_COUNT
            or consumed.get("rows_sha256") != EXPECTED_COLLECTION_ROWS_SHA256
            or consumed.get("rows_semantic_sha256")
                != EXPECTED_COLLECTION_ROWS_SEMANTIC_SHA256
            or consumed.get("identities_sha256")
                != digest(consumed.get("identities"))
            or [_identity(row, "closed v11 outer")
                for row in consumed.get("identities", [])]
                != consumed.get("identities")
            or not isinstance(projection, Mapping)
            or not isinstance(hashes, list) or hashes != sorted(set(hashes))
            or len(hashes) != UNIQUE_OBSERVATION_COUNT
            or projection.get("unique_outer_observation_hash_count") != len(hashes)
            or projection.get("outer_observation_hashes_sha256") != digest(hashes)
            or projection.get("source_projection_sha256")
                != EXPECTED_PROJECTION_SHA256
            or projection.get("source_projection_content_sha256")
                != value.get("bindings", {}).get("v11_projection_content_sha256")
            or any(projection.get(key) is not False for key in (
                "raw_observations_included", "actions_included",
                "probabilities_included", "labels_included"))
            or not isinstance(failure, Mapping)
            or failure.get("outer_consumed") is not True
            or failure.get("retry_permitted") is not False
            or failure.get("metrics_produced") is not True
            or failure.get("gate_passed") is not False
            or failure.get("failed_checks")
                != ["effective_intervention_direction_shared_charger"]
            or not isinstance(disposition, Mapping)
            or disposition.get("eligible_for_development_use") is not True
            or disposition.get("eligible_for_outer_claim") is not False
            or disposition.get("outer_identity_reuse_permitted") is not False
            or not isinstance(boundary, Mapping)
            or boundary.get("row_archive_members_opened") is not False
            or boundary.get("protected_final_access") is not False
            or boundary.get("private_salt_access") is not False
            or not _sources_valid(value.get("producer_sources"),
                                  value.get("producer_sources_sha256"))):
        raise ValueError("Saved v12 closeout semantics differ")
    permanent = _directory(permanent_attempt_registry,
                           "permanent v11 attempt registry")
    campaign = _directory(permanent / ATTEMPT_KEY,
                          "permanent consumed v11 campaign")
    anchor = _opaque(
        campaign / "attempt_anchor.json", "permanent v11 anchor",
        expected_sha256=value["bindings"]["v11_attempt_anchor_sha256"])
    result = _opaque(
        campaign / "outer_result.json", "permanent v11 result",
        expected_sha256=value["bindings"]["v11_attempt_result_sha256"])
    if (file_hash(anchor) != EXPECTED_ATTEMPT_ANCHOR_SHA256
            or file_hash(result) != EXPECTED_ATTEMPT_RESULT_SHA256
            or file_hash(receipt_file) != expected_closeout_sha256):
        raise ValueError("Saved v12 closeout permanent failure binding differs")
    return deepcopy(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "registry", "registry-report", "projection", "projection-receipt",
        "collection-rows", "collection-report", "collection-projection-copy",
        "candidate-lock", "selector-report", "program", "candidate-grid",
        "attempt-anchor", "attempt-result", "permanent-attempt-registry",
        "output",
    ):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    receipt = build(
        registry_path=args.registry,
        registry_report_path=args.registry_report,
        projection_path=args.projection,
        projection_receipt_path=args.projection_receipt,
        collection_rows_path=args.collection_rows,
        collection_report_path=args.collection_report,
        collection_projection_copy_path=args.collection_projection_copy,
        candidate_lock_path=args.candidate_lock,
        selector_report_path=args.selector_report,
        program_path=args.program,
        candidate_grid_path=args.candidate_grid,
        attempt_anchor_path=args.attempt_anchor,
        attempt_result_path=args.attempt_result,
        permanent_attempt_registry=args.permanent_attempt_registry,
        output=args.output,
    )
    print(canonical({
        "status": receipt["status"],
        "output": str(Path(args.output).resolve() / RECEIPT_NAME),
        "consumed_identity_sha256": receipt["consumed_outer"][
            "identities_sha256"],
        "rows_semantic_sha256": receipt["consumed_outer"][
            "rows_semantic_sha256"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VERSION", "STATUS", "RECEIPT_NAME", "SCENE_COUNT", "ROW_COUNT",
    "UNIQUE_OBSERVATION_COUNT", "FAMILY_COUNTS", "ATTEMPT_KEY",
    "EXPECTED_COLLECTION_ROWS_SHA256",
    "EXPECTED_COLLECTION_ROWS_SEMANTIC_SHA256", "contract",
    "producer_sources", "create_closeout", "build", "read_saved_closeout",
    "main", "_registry_identities", "_validate_chain",
]
