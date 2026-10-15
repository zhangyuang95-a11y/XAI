"""Close the consumed v9 outer attempt without reusing its observations.

The v9 one-shot scorer claimed its outer set and then failed before producing
metrics.  The claim nevertheless exposed the 64 scenes and their row payload,
so those identities can never be used as an outer set again.  This module
authenticates that complete historical chain by exact bytes, verifies the
permanent attempt anchor/result, and publishes only the public scene
identities needed to exclude the consumed set from later sampling.

No NPZ member is opened here.  In particular, this closeout never reads an
observation, action, probability, weight, or label from the consumed rows.
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


VERSION = "warehouse-r41-diagnostic-outer-attempt-closeout.v10"
STATUS = "consumed_v9_outer_irrevocably_closed"
ROOT = Path(__file__).resolve().parents[2]
RECEIPT_NAME = "closeout_receipt.json"
V9_SCENE_COUNT = 64
FAMILY_COUNTS = {
    "conflict_family_01": 11,
    "conflict_family_02": 11,
    "conflict_family_03": 11,
    "conflict_family_04": 11,
    "conflict_family_05": 10,
    "conflict_family_06": 10,
}
V9_REGISTRY_VERSION = "warehouse-r41-diagnostic-rcpd-v9-fresh-outer-registry.v1"
V9_REGISTRY_STATUS = "frozen_identity_only_pending_outer_collection"
V9_PROJECTION_VERSION = "warehouse-r41-diagnostic-outer-hash-projection.v9"
V9_PROJECTION_STATUS = "frozen_label_blind_ordered_outer_hash_projection"
V9_PROJECTION_RECEIPT_VERSION = V9_PROJECTION_VERSION + ".receipt.v1"
V9_COLLECTION_VERSION = "warehouse-r41-diagnostic-outer-collection.v9"
V9_COLLECTION_STATUS = "collected_unscored"
V9_SELECTOR_VERSION = "warehouse-r41-diagnostic-rcpd-v9-fit-selector.v1"
V9_SELECTOR_STATUS = "locked_development_candidate_pending_fresh_outer"
V9_LOCK_SCHEMA = "warehouse_r41_diagnostic_rcpd_v9_candidate_lock_v1"
V9_PROGRAM_VERSION = "warehouse-r41-diagnostic-public-tree-program.v9"
V9_GRID_VERSION = "warehouse-r41-diagnostic-rcpd-v9-candidate-grid.v1"
V9_OUTER_ONCE_VERSION = "warehouse-r41-diagnostic-rcpd-v9-outer-once.v1"
V9_ATTEMPT_KEY = "0ab89ba3701b1af5f7d1a38ed1831d272c5fb0327f4bba5e5b0e0d9224eb496d"

# These exact bytes are the already-consumed campaign.  A different artifact
# cannot silently become the source of this closeout.
EXPECTED_V9_REGISTRY_SHA256 = "adeb5347ea5b7740df9e5ce59c15f33605b9ad7e5dbadf3d544e28a8280b67d2"
EXPECTED_V9_REGISTRY_REPORT_SHA256 = "c0c0a99a4bbb4cfc4874ea50d2e572ab3027a2221b81574f1c1184385690e9ce"
EXPECTED_V9_PROJECTION_SHA256 = "594a50625af617c84edacb8491db44a37504866a945c04b0a7406f7b7254b110"
EXPECTED_V9_PROJECTION_RECEIPT_SHA256 = "557697e570d5d4a304a75636b3de5eba35a8685afb148965a93c1d00d6c8e821"
EXPECTED_V9_COLLECTION_ROWS_SHA256 = "486e337ef0eeaafd7b01c51011270575f9b8101b62a9c6c68e4acab26373b70d"
EXPECTED_V9_COLLECTION_REPORT_SHA256 = "05586e02722eaac4f39572a625e70d73a39a5d33c088361cdae2be6c93ca4fa4"
EXPECTED_V9_CANDIDATE_LOCK_SHA256 = "a31381fda303f09a7fe9b10ef64e2028689b925dfb81caeb4054178f035cf108"
EXPECTED_V9_SELECTOR_REPORT_SHA256 = "3dabb8ebf558a2d12e625a4440f81e138c7043c7b863b8a70d2c126ac986d75d"
EXPECTED_V9_PROGRAM_SHA256 = "a7b18b9838e86fed0090ce54c3eaf083d342e5c8524061635135aaf1a56a68f2"
EXPECTED_V9_CANDIDATE_GRID_SHA256 = "900578faaf4aadc4d4b0d25494d2468f654d644a3aa3d23f9b1b155dc149680e"
EXPECTED_V9_ATTEMPT_ANCHOR_SHA256 = "6c3cb84f55647df13d1d06b37f78e3fc787dceae1be1d78333c4dd7057fd06b6"
EXPECTED_V9_ATTEMPT_RESULT_SHA256 = "2848978097aebb3fde37eddc24fc18747fc93e06f16fa7f8e473fd109572a561"

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
    expected = _sha(expected_sha256, label + " SHA-256")
    if file_hash(path) != expected:
        raise ValueError("Exact " + label + " bytes required")
    return path


def _sources_valid(value: Any, claimed: Any) -> bool:
    return (isinstance(value, Mapping) and value
            and all(type(name) is str and name and type(checksum) is str
                    and _HEX.fullmatch(checksum) is not None
                    for name, checksum in value.items())
            and claimed == digest(dict(sorted(value.items()))))


def _identity(row: Any, label: str) -> dict[str, Any]:
    if not isinstance(row, Mapping) or set(row) != {
            "batch_index", "family_id", "seed", "fingerprint"}:
        raise ValueError(label + " identity fields differ")
    result = {name: row[name] for name in (
        "batch_index", "family_id", "seed", "fingerprint")}
    if (type(result["batch_index"]) is not int or result["batch_index"] < 0
            or result["family_id"] not in FAMILY_COUNTS
            or type(result["seed"]) is not int or result["seed"] < 0
            or type(result["fingerprint"]) is not str
            or _HEX.fullmatch(result["fingerprint"]) is None):
        raise ValueError(label + " identity values differ")
    return result


def _registry_identities(registry: Mapping[str, Any],
                         report: Mapping[str, Any], *, registry_sha256: str
                         ) -> list[dict[str, Any]]:
    if (registry.get("version") != V9_REGISTRY_VERSION
            or registry.get("status") != V9_REGISTRY_STATUS
            or not _content_valid(registry)
            or registry.get("program_access") is not False
            or registry.get("program_predictions_access") is not False
            or registry.get("action_labels_access") is not False
            or registry.get("probabilities_access") is not False
            or registry.get("final_audit_rows_access") is not False
            or registry.get("formal_ready") is not False
            or not _sources_valid(registry.get("producer_sources"),
                                  registry.get("producer_sources_sha256"))):
        raise ValueError("Consumed v9 registry semantics differ")
    selected, scenes = (registry.get("selected_outer_identities"),
                        registry.get("development_outer"))
    if (not isinstance(selected, list) or len(selected) != V9_SCENE_COUNT
            or not isinstance(scenes, list) or len(scenes) != V9_SCENE_COUNT):
        raise ValueError("Consumed v9 registry scene count differs")
    identities: list[dict[str, Any]] = []
    for index, (raw, scene) in enumerate(zip(selected, scenes)):
        identity = _identity(raw, "consumed v9 outer")
        if (not isinstance(scene, Mapping)
                or scene.get("id") != f"diagnostic_v9_fresh_outer_{index:04d}"
                or any(scene.get(name) != identity[name]
                       for name in ("family_id", "seed", "fingerprint"))):
            raise ValueError("Consumed v9 materialised identity differs")
        identities.append(identity)
    if (len({row["seed"] for row in identities}) != V9_SCENE_COUNT
            or len({row["fingerprint"] for row in identities}) != V9_SCENE_COUNT
            or dict(sorted(Counter(row["family_id"] for row in identities).items()))
                != FAMILY_COUNTS):
        raise ValueError("Consumed v9 outer identity population differs")
    selection = report.get("selection")
    if (report.get("version") != V9_REGISTRY_VERSION
            or report.get("status") != V9_REGISTRY_STATUS
            or not _content_valid(report)
            or report.get("formal_ready") is not False
            or report.get("registry_file_sha256") != registry_sha256
            or report.get("registry_content_sha256")
                != registry["content_sha256"]
            or report.get("bindings") != registry.get("bindings")
            or report.get("statistics") != registry.get("statistics")
            or report.get("information_boundary")
                != registry.get("information_boundary")
            or not _sources_valid(report.get("producer_sources"),
                                  report.get("producer_sources_sha256"))
            or report.get("producer_sources") != registry.get("producer_sources")
            or not isinstance(selection, Mapping)
            or selection.get("selected_identity_sha256") != digest(identities)):
        raise ValueError("Consumed v9 registry report differs")
    return identities


def _validate_chain(
    *, registry: Mapping[str, Any], registry_report: Mapping[str, Any],
    projection: Mapping[str, Any], projection_receipt: Mapping[str, Any],
    collection: Mapping[str, Any], lock: Mapping[str, Any],
    selector: Mapping[str, Any], program: Mapping[str, Any],
    grid: Mapping[str, Any], anchor: Mapping[str, Any], result: Mapping[str, Any],
    identities: Sequence[Mapping[str, Any]],
) -> None:
    identity_sha = digest(list(identities))
    if (projection.get("version") != V9_PROJECTION_VERSION
            or projection.get("status") != V9_PROJECTION_STATUS
            or not _content_valid(projection)
            or projection.get("identity", {}).get("registry_file_sha256")
                != EXPECTED_V9_REGISTRY_SHA256
            or projection.get("identity", {}).get("registry_content_sha256")
                != registry["content_sha256"]
            or projection.get("identity", {}).get("selected_identity_sha256")
                != identity_sha):
        raise ValueError("Consumed v9 label-blind projection differs")
    projection_data = projection.get("projection")
    if (not isinstance(projection_data, Mapping)
            or projection_data.get("row_count") != 35733
            or type(projection_data.get("ordered_observation_hashes")) is not list
            or len(projection_data["ordered_observation_hashes"]) != 35733
            or projection_data.get("ordered_observation_hashes_sha256")
                != digest(projection_data["ordered_observation_hashes"])):
        raise ValueError("Consumed v9 projection row commitment differs")
    receipt_bindings = projection_receipt.get("bindings")
    if (projection_receipt.get("version") != V9_PROJECTION_RECEIPT_VERSION
            or projection_receipt.get("status") != V9_PROJECTION_STATUS
            or not _content_valid(projection_receipt)
            or not isinstance(receipt_bindings, Mapping)
            or receipt_bindings.get("fresh_outer_registry_sha256")
                != EXPECTED_V9_REGISTRY_SHA256
            or receipt_bindings.get("fresh_outer_registry_report_sha256")
                != EXPECTED_V9_REGISTRY_REPORT_SHA256
            or receipt_bindings.get("outer_hash_projection_sha256")
                != EXPECTED_V9_PROJECTION_SHA256
            or receipt_bindings.get("outer_hash_projection_content_sha256")
                != projection["content_sha256"]):
        raise ValueError("Consumed v9 projection receipt differs")
    lock_bindings = lock.get("bindings")
    if (lock.get("schema_version") != V9_LOCK_SCHEMA
            or lock.get("status") != "locked" or not _content_valid(lock)
            or lock.get("formal_ready") is not False
            or not isinstance(lock_bindings, Mapping)
            or not _sources_valid(lock.get("source_closure"),
                                  lock_bindings.get("source_closure_sha256"))
            or lock_bindings.get("fresh_outer_registry_sha256")
                != EXPECTED_V9_REGISTRY_SHA256
            or lock_bindings.get("fresh_outer_registry_report_sha256")
                != EXPECTED_V9_REGISTRY_REPORT_SHA256
            or lock_bindings.get("outer_hash_projection_sha256")
                != EXPECTED_V9_PROJECTION_SHA256
            or lock_bindings.get("outer_hash_projection_receipt_sha256")
                != EXPECTED_V9_PROJECTION_RECEIPT_SHA256
            or lock_bindings.get("selector_report_sha256")
                != EXPECTED_V9_SELECTOR_REPORT_SHA256
            or lock_bindings.get("program_sha256") != EXPECTED_V9_PROGRAM_SHA256
            or lock_bindings.get("candidate_grid_sha256")
                != EXPECTED_V9_CANDIDATE_GRID_SHA256
            or lock.get("selection", {}).get("fresh_outer_scored") is not False
            or lock.get("selection", {}).get(
                "two_salt_three_fold_development_cv_passed") is not True):
        raise ValueError("Consumed v9 selector lock differs")
    if (selector.get("version") != V9_SELECTOR_VERSION
            or selector.get("status") != V9_SELECTOR_STATUS
            or not _content_valid(selector)
            or selector.get("formal_ready") is not False
            or selector.get("bindings", {}).get("program_sha256")
                != EXPECTED_V9_PROGRAM_SHA256
            or selector.get("bindings", {}).get("candidate_grid_sha256")
                != EXPECTED_V9_CANDIDATE_GRID_SHA256
            or selector.get("bindings", {}).get("fresh_outer_registry_sha256")
                != EXPECTED_V9_REGISTRY_SHA256
            or selector.get("bindings", {}).get("outer_hash_projection_sha256")
                != EXPECTED_V9_PROJECTION_SHA256
            or selector.get("selection", {}).get("selected_config_sha256")
                != lock.get("selection", {}).get("selected_config_sha256")
            or selector.get("final_fit", {}).get("program_file_sha256")
                != EXPECTED_V9_PROGRAM_SHA256):
        raise ValueError("Consumed v9 selector report differs")
    if (program.get("version") != V9_PROGRAM_VERSION
            or program.get("metadata", {}).get("fit_config_sha256")
                != selector.get("selection", {}).get("selected_config_sha256")
            or program.get("metadata", {}).get("runtime_action_override") is not False
            or grid.get("version") != V9_GRID_VERSION
            or not _content_valid(grid)):
        raise ValueError("Consumed v9 selector program or grid differs")
    collection_bindings = collection.get("bindings")
    collection_state = collection.get("collection")
    if (collection.get("version") != V9_COLLECTION_VERSION
            or collection.get("status") != V9_COLLECTION_STATUS
            or not _content_valid(collection)
            or collection.get("formal_ready") is not False
            or not isinstance(collection_bindings, Mapping)
            or collection_bindings.get("fresh_outer_registry_sha256")
                != EXPECTED_V9_REGISTRY_SHA256
            or collection_bindings.get("fresh_outer_registry_report_sha256")
                != EXPECTED_V9_REGISTRY_REPORT_SHA256
            or collection_bindings.get("outer_hash_projection_sha256")
                != EXPECTED_V9_PROJECTION_SHA256
            or collection_bindings.get("outer_hash_projection_receipt_sha256")
                != EXPECTED_V9_PROJECTION_RECEIPT_SHA256
            or collection_bindings.get("candidate_lock_sha256")
                != EXPECTED_V9_CANDIDATE_LOCK_SHA256
            or collection_bindings.get("selector_report_sha256")
                != EXPECTED_V9_SELECTOR_REPORT_SHA256
            or collection_bindings.get("program_sha256")
                != EXPECTED_V9_PROGRAM_SHA256
            or collection_bindings.get("rows_sha256")
                != EXPECTED_V9_COLLECTION_ROWS_SHA256
            or collection_bindings.get("projection_copy_sha256")
                != EXPECTED_V9_PROJECTION_SHA256
            or not isinstance(collection_state, Mapping)
            or collection_state.get("scene_count") != V9_SCENE_COUNT
            or collection_state.get("row_count") != 35733
            or collection_state.get("scored") is not False
            or collection_state.get("score_computed") is not False
            or collection_state.get(
                "all_submitted_actions_equal_policy_actions") is not True):
        raise ValueError("Consumed v9 unscored collection differs")
    attempt_inputs, anchor_bindings = (anchor.get("attempt_key_inputs"),
                                       anchor.get("bindings"))
    if (anchor.get("version") != V9_OUTER_ONCE_VERSION + ".attempt-anchor.v1"
            or anchor.get("status") != "fresh_outer_attempt_irrevocably_claimed"
            or anchor.get("attempt_key") != V9_ATTEMPT_KEY
            or not _content_valid(anchor)
            or anchor.get("retry_permitted") is not False
            or anchor.get("private_outer_fields_read_before_claim") is not False
            or anchor.get("protected_final_access") is not False
            or not isinstance(attempt_inputs, Mapping)
            or digest(dict(attempt_inputs)) != V9_ATTEMPT_KEY
            or attempt_inputs.get("fresh_outer_registry_sha256")
                != EXPECTED_V9_REGISTRY_SHA256
            or attempt_inputs.get("fresh_outer_registry_report_sha256")
                != EXPECTED_V9_REGISTRY_REPORT_SHA256
            or attempt_inputs.get("outer_hash_projection_sha256")
                != EXPECTED_V9_PROJECTION_SHA256
            or attempt_inputs.get("selected_identity_sha256") != identity_sha
            or not isinstance(anchor_bindings, Mapping)
            or anchor_bindings.get("candidate_lock_sha256")
                != EXPECTED_V9_CANDIDATE_LOCK_SHA256
            or anchor_bindings.get("program_sha256") != EXPECTED_V9_PROGRAM_SHA256
            or anchor_bindings.get("outer_collection_report_sha256")
                != EXPECTED_V9_COLLECTION_REPORT_SHA256
            or anchor_bindings.get("outer_rows_sha256")
                != EXPECTED_V9_COLLECTION_ROWS_SHA256
            or anchor_bindings.get("fresh_outer_registry_report_sha256")
                != EXPECTED_V9_REGISTRY_REPORT_SHA256
            or anchor_bindings.get("selected_identity_sha256") != identity_sha
            or anchor.get("preflight", {}).get("outer_scene_count")
                != V9_SCENE_COUNT
            or anchor.get("preflight", {}).get("outer_row_count") != 35733):
        raise ValueError("Consumed v9 permanent attempt anchor differs")
    failure = result.get("failure")
    if (result.get("version") != V9_OUTER_ONCE_VERSION
            or result.get("status") != "failed_after_outer_attempt_claim"
            or result.get("attempt_key") != V9_ATTEMPT_KEY
            or not _content_valid(result)
            or result.get("attempt_anchor_content_sha256")
                != anchor["content_sha256"]
            or result.get("formal_ready") is not False
            or result.get("protected_final_access") is not False
            or not isinstance(failure, Mapping)
            or failure.get("outer_consumed") is not True
            or failure.get("retry_permitted") is not False):
        raise ValueError("Consumed v9 permanent attempt result differs")


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "source": "authenticated failed-after-claim v9 one-shot outer",
        "consumed_scene_count": V9_SCENE_COUNT,
        "consumed_outer_is_development_exposed": True,
        "outer_identity_reuse_permitted": False,
        "rows_opened": False,
        "observations_actions_probabilities_or_labels_read": False,
        "one_way_observation_hash_projection_published": True,
        "protected_final_access": False,
        "formal_ready": False,
    }


def create_closeout(
    *, v9_registry_path: str | Path, v9_registry_report_path: str | Path,
    v9_projection_path: str | Path, v9_projection_receipt_path: str | Path,
    v9_collection_rows_path: str | Path,
    v9_collection_report_path: str | Path,
    v9_collection_projection_copy_path: str | Path,
    v9_candidate_lock_path: str | Path, v9_selector_report_path: str | Path,
    v9_program_path: str | Path, v9_candidate_grid_path: str | Path,
    v9_attempt_anchor_path: str | Path, v9_attempt_result_path: str | Path,
    permanent_attempt_registry: str | Path,
) -> dict[str, Any]:
    sources = producer_sources()
    registry_path, _, registry = _strict_json(
        v9_registry_path, "consumed v9 registry",
        expected_sha256=EXPECTED_V9_REGISTRY_SHA256)
    registry_report_path, _, registry_report = _strict_json(
        v9_registry_report_path, "consumed v9 registry report",
        expected_sha256=EXPECTED_V9_REGISTRY_REPORT_SHA256)
    identities = _registry_identities(
        registry, registry_report, registry_sha256=file_hash(registry_path))
    projection_path, projection_raw, projection = _strict_json(
        v9_projection_path, "consumed v9 outer hash projection",
        expected_sha256=EXPECTED_V9_PROJECTION_SHA256)
    projection_receipt_path, _, projection_receipt = _strict_json(
        v9_projection_receipt_path, "consumed v9 projection receipt",
        expected_sha256=EXPECTED_V9_PROJECTION_RECEIPT_SHA256)
    rows_path = _opaque(
        v9_collection_rows_path, "consumed v9 row archive",
        expected_sha256=EXPECTED_V9_COLLECTION_ROWS_SHA256)
    collection_report_path, _, collection = _strict_json(
        v9_collection_report_path, "consumed v9 collection report",
        expected_sha256=EXPECTED_V9_COLLECTION_REPORT_SHA256)
    projection_copy_path = _opaque(
        v9_collection_projection_copy_path, "consumed v9 projection copy",
        expected_sha256=EXPECTED_V9_PROJECTION_SHA256)
    if projection_copy_path.read_bytes() != projection_raw:
        raise ValueError("Consumed collection projection copy differs")
    lock_path, _, lock = _strict_json(
        v9_candidate_lock_path, "consumed v9 candidate lock",
        expected_sha256=EXPECTED_V9_CANDIDATE_LOCK_SHA256)
    selector_path, _, selector = _strict_json(
        v9_selector_report_path, "consumed v9 selector report",
        expected_sha256=EXPECTED_V9_SELECTOR_REPORT_SHA256)
    program_path, _, program = _strict_json(
        v9_program_path, "consumed v9 program",
        expected_sha256=EXPECTED_V9_PROGRAM_SHA256)
    grid_path, _, grid = _strict_json(
        v9_candidate_grid_path, "consumed v9 candidate grid",
        expected_sha256=EXPECTED_V9_CANDIDATE_GRID_SHA256)
    permanent = _directory(permanent_attempt_registry,
                           "permanent v9 attempt registry")
    campaign = _directory(permanent / V9_ATTEMPT_KEY,
                          "permanent consumed v9 campaign")
    if {entry.name for entry in campaign.iterdir()} != {
            "attempt_anchor.json", "outer_result.json"}:
        raise ValueError("Permanent consumed v9 campaign entries differ")
    expected_anchor = campaign / "attempt_anchor.json"
    expected_result = campaign / "outer_result.json"
    if (Path(v9_attempt_anchor_path).expanduser().absolute().resolve()
            != expected_anchor
            or Path(v9_attempt_result_path).expanduser().absolute().resolve()
            != expected_result):
        raise ValueError("Attempt files must be the permanent campaign files")
    anchor_path, _, anchor = _strict_json(
        expected_anchor, "permanent consumed v9 attempt anchor",
        expected_sha256=EXPECTED_V9_ATTEMPT_ANCHOR_SHA256)
    result_path, _, result = _strict_json(
        expected_result, "permanent consumed v9 attempt result",
        expected_sha256=EXPECTED_V9_ATTEMPT_RESULT_SHA256)
    _validate_chain(
        registry=registry, registry_report=registry_report,
        projection=projection, projection_receipt=projection_receipt,
        collection=collection, lock=lock, selector=selector,
        program=program, grid=grid, anchor=anchor, result=result,
        identities=identities)
    projection_data = projection["projection"]
    bindings = {
        "v9_registry_sha256": file_hash(registry_path),
        "v9_registry_content_sha256": registry["content_sha256"],
        "v9_registry_report_sha256": file_hash(registry_report_path),
        "v9_registry_report_content_sha256": registry_report["content_sha256"],
        "v9_projection_sha256": file_hash(projection_path),
        "v9_projection_content_sha256": projection["content_sha256"],
        "v9_projection_receipt_sha256": file_hash(projection_receipt_path),
        "v9_collection_rows_sha256": file_hash(rows_path),
        "v9_collection_report_sha256": file_hash(collection_report_path),
        "v9_collection_report_content_sha256": collection["content_sha256"],
        "v9_collection_projection_copy_sha256": file_hash(projection_copy_path),
        "v9_candidate_lock_sha256": file_hash(lock_path),
        "v9_candidate_lock_content_sha256": lock["content_sha256"],
        "v9_selector_report_sha256": file_hash(selector_path),
        "v9_selector_report_content_sha256": selector["content_sha256"],
        "v9_program_sha256": file_hash(program_path),
        "v9_candidate_grid_sha256": file_hash(grid_path),
        "v9_attempt_anchor_sha256": file_hash(anchor_path),
        "v9_attempt_anchor_content_sha256": anchor["content_sha256"],
        "v9_attempt_result_sha256": file_hash(result_path),
        "v9_attempt_result_content_sha256": result["content_sha256"],
        "contract_sha256": digest(contract()),
    }
    receipt: dict[str, Any] = {
        "version": VERSION,
        "status": STATUS,
        "attempt_key": V9_ATTEMPT_KEY,
        "contract": contract(),
        "bindings": bindings,
        "consumed_outer": {
            "scene_count": V9_SCENE_COUNT,
            "identities": deepcopy(list(identities)),
            "identities_sha256": digest(list(identities)),
            "family_counts": deepcopy(FAMILY_COUNTS),
            "row_count": 35733,
            "rows_sha256": EXPECTED_V9_COLLECTION_ROWS_SHA256,
            "observation_hash_projection": {
                "source_projection_sha256": EXPECTED_V9_PROJECTION_SHA256,
                "source_projection_content_sha256": projection[
                    "content_sha256"],
                "outer_observation_hashes": deepcopy(
                    projection_data["unique_observation_hashes"]),
                "unique_outer_observation_hash_count": projection_data[
                    "unique_observation_count"],
                "outer_observation_hashes_sha256": projection_data[
                    "unique_observation_hashes_sha256"],
                "raw_observations_included": False,
                "actions_included": False,
                "probabilities_included": False,
                "labels_included": False,
            },
        },
        "failure": {
            "status": result["status"],
            "exception_type": result["failure"].get("exception_type"),
            "message": result["failure"].get("message"),
            "outer_consumed": True,
            "retry_permitted": False,
            "metrics_produced": False,
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
            "row_archive_members_opened": False,
            "raw_observations_read": False,
            "actions_probabilities_weights_or_labels_read": False,
            "program_executed": False,
            "protected_final_access": False,
            "formal_ready": False,
        },
        "producer_sources": sources,
        "producer_sources_sha256": digest(sources),
        "formal_ready": False,
    }
    receipt["content_sha256"] = digest(receipt)
    if producer_sources() != sources:
        raise RuntimeError("v10 closeout source closure changed during build")
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
    parent = _directory(destination.parent, "v10 closeout output parent")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("v10 closeout output already exists")
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
    receipt_path, _, value = _strict_json(
        path, "v10 consumed-v9 closeout",
        expected_sha256=expected_closeout_sha256)
    consumed, failure, disposition = (value.get("consumed_outer"),
                                       value.get("failure"),
                                       value.get("disposition"))
    projected = consumed.get("observation_hash_projection") \
        if isinstance(consumed, Mapping) else None
    projected_hashes = projected.get("outer_observation_hashes") \
        if isinstance(projected, Mapping) else None
    if (value.get("version") != VERSION or value.get("status") != STATUS
            or not _content_valid(value) or value.get("formal_ready") is not False
            or value.get("contract") != contract()
            or not isinstance(consumed, Mapping)
            or consumed.get("scene_count") != V9_SCENE_COUNT
            or consumed.get("identities_sha256")
                != digest(consumed.get("identities"))
            or [_identity(row, "closed v9 outer")
                for row in consumed.get("identities", [])]
                != consumed.get("identities")
            or not isinstance(projected, Mapping)
            or not isinstance(projected_hashes, list)
            or projected_hashes != sorted(set(projected_hashes))
            or len(projected_hashes) != 30776
            or any(type(item) is not str or _HEX.fullmatch(item) is None
                   for item in projected_hashes)
            or projected.get("unique_outer_observation_hash_count")
                != len(projected_hashes)
            or projected.get("outer_observation_hashes_sha256")
                != digest(projected_hashes)
            or projected.get("source_projection_sha256")
                != EXPECTED_V9_PROJECTION_SHA256
            or projected.get("raw_observations_included") is not False
            or projected.get("actions_included") is not False
            or projected.get("probabilities_included") is not False
            or projected.get("labels_included") is not False
            or not isinstance(failure, Mapping)
            or failure.get("outer_consumed") is not True
            or failure.get("retry_permitted") is not False
            or failure.get("metrics_produced") is not False
            or not isinstance(disposition, Mapping)
            or disposition.get("eligible_for_outer_claim") is not False
            or disposition.get("outer_identity_reuse_permitted") is not False
            or not _sources_valid(value.get("producer_sources"),
                                  value.get("producer_sources_sha256"))):
        raise ValueError("Saved v10 closeout semantics differ")
    permanent = _directory(permanent_attempt_registry,
                           "permanent v9 attempt registry")
    campaign = _directory(permanent / V9_ATTEMPT_KEY,
                          "permanent consumed v9 campaign")
    anchor = _opaque(campaign / "attempt_anchor.json", "permanent v9 anchor",
                     expected_sha256=value["bindings"]["v9_attempt_anchor_sha256"])
    result = _opaque(campaign / "outer_result.json", "permanent v9 result",
                     expected_sha256=value["bindings"]["v9_attempt_result_sha256"])
    if (file_hash(anchor) != EXPECTED_V9_ATTEMPT_ANCHOR_SHA256
            or file_hash(result) != EXPECTED_V9_ATTEMPT_RESULT_SHA256
            or file_hash(receipt_path) != expected_closeout_sha256):
        raise ValueError("Saved v10 closeout permanent failure binding differs")
    return deepcopy(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v9-registry", required=True)
    parser.add_argument("--v9-registry-report", required=True)
    parser.add_argument("--v9-projection", required=True)
    parser.add_argument("--v9-projection-receipt", required=True)
    parser.add_argument("--v9-collection-rows", required=True)
    parser.add_argument("--v9-collection-report", required=True)
    parser.add_argument("--v9-collection-projection-copy", required=True)
    parser.add_argument("--v9-candidate-lock", required=True)
    parser.add_argument("--v9-selector-report", required=True)
    parser.add_argument("--v9-program", required=True)
    parser.add_argument("--v9-candidate-grid", required=True)
    parser.add_argument("--v9-attempt-anchor", required=True)
    parser.add_argument("--v9-attempt-result", required=True)
    parser.add_argument("--permanent-attempt-registry", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    receipt = build(
        v9_registry_path=args.v9_registry,
        v9_registry_report_path=args.v9_registry_report,
        v9_projection_path=args.v9_projection,
        v9_projection_receipt_path=args.v9_projection_receipt,
        v9_collection_rows_path=args.v9_collection_rows,
        v9_collection_report_path=args.v9_collection_report,
        v9_collection_projection_copy_path=args.v9_collection_projection_copy,
        v9_candidate_lock_path=args.v9_candidate_lock,
        v9_selector_report_path=args.v9_selector_report,
        v9_program_path=args.v9_program,
        v9_candidate_grid_path=args.v9_candidate_grid,
        v9_attempt_anchor_path=args.v9_attempt_anchor,
        v9_attempt_result_path=args.v9_attempt_result,
        permanent_attempt_registry=args.permanent_attempt_registry,
        output=args.output,
    )
    print(canonical({
        "status": receipt["status"],
        "output": str(Path(args.output).resolve() / RECEIPT_NAME),
        "consumed_identity_sha256": receipt["consumed_outer"][
            "identities_sha256"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VERSION", "STATUS", "RECEIPT_NAME", "V9_SCENE_COUNT", "FAMILY_COUNTS",
    "contract", "producer_sources", "create_closeout", "build",
    "read_saved_closeout", "main", "_registry_identities", "_validate_chain",
]
