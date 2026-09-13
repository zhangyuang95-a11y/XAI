"""Close the burned v10 final attempt and expose only its prior-outer roots.

The v10 protected-final controller made its permanent claim and then failed
inside the historical materializer's public-input authentication.  This
closeout authenticates the immutable claim/completion pair, the passed v10
outer, and the exact historical materializer source.  It reproduces the
public observation-hash overlap that made the historical materializer raise
before its salt-read call.  The private salt path is treated as an opaque
string; this module never stats or opens it.

The receipt makes the passed v10 outer development-exposed.  Its 64 scene
identities and its one-way public observation-hash projection must therefore
be excluded by every replacement outer/final campaign.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np

from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import canonical, digest, file_hash


VERSION = "warehouse-r41-diagnostic-final-attempt-closeout.v11"
STATUS = "burned_v10_final_pre_secret_irrevocably_closed"
RECEIPT_NAME = "closeout_receipt.json"
ROOT = Path(__file__).resolve().parents[2]
FINAL_ATTEMPT_KEY = "fab70a525a8af9c60204c9542b2be0cde9c9d0d558877f500d090642da0233ae"
OUTER_ATTEMPT_KEY = "2f7eee0f60077e686c28070125cbffbf3c3bb3bd77bdf9ea9e9d00c96281399d"
FAMILY_COUNTS = {
    "conflict_family_01": 11, "conflict_family_02": 11,
    "conflict_family_03": 11, "conflict_family_04": 11,
    "conflict_family_05": 10, "conflict_family_06": 10,
}
SCENE_COUNT = 64
EXPECTED_OVERLAP_UNIQUE_HASHES = 1505
EXPECTED_OVERLAP_ROWS = 1902

# Exact public bytes consumed by the v10 campaign.
EXPECTED_REGISTRY_SHA256 = "83ff555aa75942e5fff8d6fbbf9083a0793faf676af04e9ded65eb3b1796251e"
EXPECTED_REGISTRY_REPORT_SHA256 = "620f05213381338915a741c3e54914c274078fb29614ffa820bbb41472653e4c"
EXPECTED_PROJECTION_SHA256 = "726eb10325d9372777b5cb9f0ec9bb8043aad2284835e50dd1e4b3c46314269f"
EXPECTED_PROJECTION_RECEIPT_SHA256 = "d6f82f3b1fb48d12a070efe04cee6e14647ac34cb6beb9ab34b3ca222a5c4bb8"
EXPECTED_COLLECTION_ROWS_SHA256 = "1bd90efdb6bc38ffc284f8235d8ee5a18ddad5ac14027e70bdea7c5bdc6b66f1"
EXPECTED_COLLECTION_REPORT_SHA256 = "e92a632c6a7a4fb6094094acdf2b0d8a00ba0c41cd28a2e0c8b16b05a91f609e"
EXPECTED_CANDIDATE_LOCK_SHA256 = "0a379445223c35351a42ce8887933b9bd46aa243047f1dccbca2f15747d342a7"
EXPECTED_SELECTOR_REPORT_SHA256 = "a4974094329f9e93e9b5e9f90690adcfc2f161b37c5efabdcfa0fefaa7de4b3b"
EXPECTED_PROGRAM_SHA256 = "d1a31894b638ae8eeefd64cbae3d4a122a2c6ff8f262b73cc84950faf46d9ad8"
EXPECTED_OUTER_ANCHOR_SHA256 = "04848eadd4f94e107e9cb6a074d05a8a23cd1dea3f5948c363ea5d989c04707e"
EXPECTED_OUTER_RESULT_SHA256 = "b06e0f76c4aa7d27f8ffa136cd0db06c106af53f043760da8b8cf7181d0d3b93"
EXPECTED_FINAL_ANCHOR_SHA256 = "d5477c8c6b3d934d0c52cae6002caffd411151e57aad1a5f25abf541191ee3fd"
EXPECTED_FINAL_COMPLETION_SHA256 = "7b187b8ab0daaaf43b173e4527111d2097010773f1b7ca4596f6b7fbfc42f1c0"
EXPECTED_CONFIG_SHA256 = "1790cb31659141ec59ab998d9a601b567175e05413091e3ef9aff6592b14aba3"
EXPECTED_DEVELOPMENT_ROWS_SHA256 = "cf631ef9ddd16f333d0cd43121650633eb58a337e473bb8279d9ab12ace3024f"

# The claim binds this historical source closure.  Commit 580252a is the
# public, immutable source snapshot that was present when the attempt ran.
HISTORICAL_COMMIT = "580252a5c07034bfb23949b7b6947c84e236f72d"
HISTORICAL_MATERIALIZER = (
    "backend/training/warehouse_r41_diagnostic_final_materializer_v9.py")
HISTORICAL_MATERIALIZER_SHA256 = "2bc62a6a4e97b7f3f41810e35a526b0ad5b45391521b30c21ab388c5696fbf20"
HISTORICAL_MATERIALIZER_CLOSURE_SHA256 = "6924a7982159c50697835c2bf0755737c1f8235551d713e4c9491e30e2b9fec9"
HISTORICAL_CONTROLLER_CLOSURE_SHA256 = "59af3f47df98cd11e718956ec8897888fd290d872b32a3527ff09451635e350e"

MAX_JSON_BYTES = 512 * 1024 * 1024
MAX_NPZ_BYTES = 2 * 1024 * 1024 * 1024
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def producer_sources() -> dict[str, str]:
    return dict(sorted(local_source_hashes((Path(__file__).resolve(),)).items()))


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


def _regular(value: str | Path, label: str, *, maximum: int,
             expected_sha256: str) -> Path:
    path = Path(value).expanduser().absolute()
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or path.stat(follow_symlinks=False).st_size <= 0
            or path.stat(follow_symlinks=False).st_size > maximum
            or file_hash(path) != expected_sha256):
        raise ValueError("Exact " + label + " bytes required")
    return path


def _strict_json(value: str | Path, label: str, *, expected_sha256: str
                 ) -> tuple[Path, dict[str, Any]]:
    path = _regular(value, label, maximum=MAX_JSON_BYTES,
                    expected_sha256=expected_sha256)
    raw = path.read_bytes()

    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in rows:
            if key in result:
                raise ValueError("Duplicate JSON field in " + label)
            result[key] = child
        return result

    parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                        parse_constant=lambda token: (_ for _ in ()).throw(
                            ValueError("Non-finite JSON in " + label)))
    if not isinstance(parsed, dict) or file_hash(path) != expected_sha256:
        raise ValueError(label + " changed during read")
    return path, parsed


def _historical_materializer_bytes() -> bytes:
    completed = subprocess.run(
        ["git", "show", HISTORICAL_COMMIT + ":" + HISTORICAL_MATERIALIZER],
        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, check=True)
    raw = completed.stdout
    if sha256(raw).hexdigest() != HISTORICAL_MATERIALIZER_SHA256:
        raise ValueError("Historical final materializer source differs")
    return raw


def _call_order(raw: bytes) -> list[str]:
    tree = ast.parse(raw.decode("utf-8"), filename=HISTORICAL_MATERIALIZER)
    function = next((node for node in tree.body
                     if isinstance(node, ast.FunctionDef)
                     and node.name == "materialize"), None)
    if function is None:
        raise ValueError("Historical materialize function is missing")
    result: list[str] = []
    for statement in function.body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                result.append(node.func.id)
                break
    return result


def _project_hashes(rows_path: Path) -> list[str]:
    try:
        with zipfile.ZipFile(rows_path) as archive:
            info = archive.getinfo("observation_hashes.npy")
            if info.is_dir() or info.file_size <= 0 or info.file_size > MAX_JSON_BYTES:
                raise ValueError("Development observation-hash member is unsafe")
            array = np.load(BytesIO(archive.read(info)), allow_pickle=False)
    except (OSError, KeyError, zipfile.BadZipFile) as error:
        raise ValueError("Development hash projection cannot be read") from error
    if array.ndim != 1 or array.dtype.kind != "S":
        raise ValueError("Development hashes must be one byte-string vector")
    return list(map(str, np.char.decode(array, "ascii")))


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "source": "authenticated burned v10 protected-final attempt",
        "final_attempt_consumed": True,
        "retry_same_candidate_outer_permitted": False,
        "failure_proven_before_private_salt_read": True,
        "protected_final_identity_revealed": False,
        "v10_outer_is_development_exposed": True,
        "v10_outer_identity_reuse_permitted": False,
        "v10_outer_observation_hash_reuse_permitted": False,
        "private_salt_stat_or_read": False,
        "formal_ready": False,
    }


def create_closeout(
    *, v10_registry_path: str | Path, v10_registry_report_path: str | Path,
    v10_projection_path: str | Path,
    v10_projection_receipt_path: str | Path,
    v10_collection_rows_path: str | Path,
    v10_collection_report_path: str | Path,
    v10_collection_projection_copy_path: str | Path,
    v10_candidate_lock_path: str | Path,
    v10_selector_report_path: str | Path, v10_program_path: str | Path,
    v10_outer_anchor_path: str | Path, v10_outer_result_path: str | Path,
    permanent_v10_outer_registry: str | Path,
    v10_final_anchor_path: str | Path, v10_final_completion_path: str | Path,
    permanent_v10_final_registry: str | Path,
    materializer_config_path: str | Path,
    development_rows_path: str | Path,
) -> dict[str, Any]:
    sources = producer_sources()
    registry_path, registry = _strict_json(
        v10_registry_path, "v10 outer registry",
        expected_sha256=EXPECTED_REGISTRY_SHA256)
    report_path, report = _strict_json(
        v10_registry_report_path, "v10 outer registry report",
        expected_sha256=EXPECTED_REGISTRY_REPORT_SHA256)
    projection_path, projection = _strict_json(
        v10_projection_path, "v10 outer projection",
        expected_sha256=EXPECTED_PROJECTION_SHA256)
    projection_receipt_path, projection_receipt = _strict_json(
        v10_projection_receipt_path, "v10 projection receipt",
        expected_sha256=EXPECTED_PROJECTION_RECEIPT_SHA256)
    rows_path = _regular(
        v10_collection_rows_path, "v10 collection rows", maximum=MAX_NPZ_BYTES,
        expected_sha256=EXPECTED_COLLECTION_ROWS_SHA256)
    collection_path, collection = _strict_json(
        v10_collection_report_path, "v10 collection report",
        expected_sha256=EXPECTED_COLLECTION_REPORT_SHA256)
    projection_copy = _regular(
        v10_collection_projection_copy_path, "v10 collection projection copy",
        maximum=MAX_JSON_BYTES, expected_sha256=EXPECTED_PROJECTION_SHA256)
    lock_path, lock = _strict_json(
        v10_candidate_lock_path, "v10 candidate lock",
        expected_sha256=EXPECTED_CANDIDATE_LOCK_SHA256)
    selector_path, selector = _strict_json(
        v10_selector_report_path, "v10 selector report",
        expected_sha256=EXPECTED_SELECTOR_REPORT_SHA256)
    program_path, program = _strict_json(
        v10_program_path, "v10 program", expected_sha256=EXPECTED_PROGRAM_SHA256)
    outer_anchor_path, outer_anchor = _strict_json(
        v10_outer_anchor_path, "v10 outer anchor",
        expected_sha256=EXPECTED_OUTER_ANCHOR_SHA256)
    outer_result_path, outer_result = _strict_json(
        v10_outer_result_path, "v10 outer result",
        expected_sha256=EXPECTED_OUTER_RESULT_SHA256)
    final_anchor_path, final_anchor = _strict_json(
        v10_final_anchor_path, "burned v10 final anchor",
        expected_sha256=EXPECTED_FINAL_ANCHOR_SHA256)
    final_completion_path, final_completion = _strict_json(
        v10_final_completion_path, "burned v10 final completion",
        expected_sha256=EXPECTED_FINAL_COMPLETION_SHA256)
    config_path, config = _strict_json(
        materializer_config_path, "burned v10 materializer config",
        expected_sha256=EXPECTED_CONFIG_SHA256)
    development_path = _regular(
        development_rows_path, "v10 development rows", maximum=MAX_NPZ_BYTES,
        expected_sha256=EXPECTED_DEVELOPMENT_ROWS_SHA256)

    outer_permanent = _directory(
        permanent_v10_outer_registry, "permanent v10 outer registry")
    outer_campaign = _directory(
        outer_permanent / OUTER_ATTEMPT_KEY, "permanent v10 outer campaign")
    if (outer_anchor_path.resolve() != (outer_campaign / "attempt_anchor.json")
            or outer_result_path.resolve() != (outer_campaign / "outer_result.json")):
        raise ValueError("v10 outer files are not permanent campaign files")
    final_permanent = _directory(
        permanent_v10_final_registry, "permanent v10 final registry")
    final_campaign = _directory(
        final_permanent / FINAL_ATTEMPT_KEY, "permanent v10 final campaign")
    if (final_anchor_path.resolve() != (final_campaign / "attempt_anchor.json")
            or final_completion_path.resolve()
                != (final_campaign / "attempt_completed.json")
            or {entry.name for entry in final_campaign.iterdir()} != {
                "attempt_anchor.json", "attempt_completed.json"}):
        raise ValueError("Burned final campaign entries differ")

    identities = registry.get("selected_outer_identities")
    scenes = registry.get("development_outer")
    if (not _content_valid(registry) or not _content_valid(report)
            or not isinstance(identities, list) or len(identities) != SCENE_COUNT
            or not isinstance(scenes, list) or len(scenes) != SCENE_COUNT
            or report.get("registry_file_sha256") != EXPECTED_REGISTRY_SHA256
            or report.get("registry_content_sha256") != registry["content_sha256"]
            or report.get("selection", {}).get("selected_identity_sha256")
                != digest(identities)):
        raise ValueError("v10 outer registry chain differs")
    public: list[dict[str, Any]] = []
    for index, (identity, scene) in enumerate(zip(identities, scenes)):
        if not isinstance(identity, Mapping):
            raise ValueError("v10 outer identity differs")
        row = {name: identity.get(name) for name in (
            "batch_index", "family_id", "seed", "fingerprint")}
        if (type(row["batch_index"]) is not int
                or row["family_id"] not in FAMILY_COUNTS
                or type(row["seed"]) is not int
                or type(row["fingerprint"]) is not str
                or _HEX.fullmatch(row["fingerprint"]) is None
                or not isinstance(scene, Mapping)
                or scene.get("id") != f"diagnostic_v10_fresh_outer_{index:04d}"
                or any(scene.get(name) != row[name] for name in (
                    "family_id", "seed", "fingerprint"))):
            raise ValueError("v10 outer materialised identity differs")
        public.append(row)
    if (dict(sorted(Counter(row["family_id"] for row in public).items()))
            != FAMILY_COUNTS or len({row["seed"] for row in public}) != SCENE_COUNT
            or len({row["fingerprint"] for row in public}) != SCENE_COUNT):
        raise ValueError("v10 outer identity accounting differs")

    projection_data = projection.get("projection")
    unique_hashes = projection_data.get("unique_observation_hashes") \
        if isinstance(projection_data, Mapping) else None
    if (not _content_valid(projection)
            or not isinstance(unique_hashes, list)
            or unique_hashes != sorted(set(unique_hashes))
            or len(unique_hashes) != 31150
            or any(type(value) is not str or _HEX.fullmatch(value) is None
                   for value in unique_hashes)
            or projection_data.get("unique_observation_hashes_sha256")
                != digest(unique_hashes)
            or projection.get("identity", {}).get("registry_file_sha256")
                != EXPECTED_REGISTRY_SHA256
            or projection.get("identity", {}).get("selected_identity_sha256")
                != digest(public)
            or not _content_valid(projection_receipt)
            or projection_receipt.get("bindings", {}).get(
                "outer_hash_projection_sha256") != EXPECTED_PROJECTION_SHA256
            or projection_copy.read_bytes() != projection_path.read_bytes()):
        raise ValueError("v10 outer projection chain differs")

    lock_bindings = lock.get("bindings", {})
    if (not _content_valid(lock) or not _content_valid(selector)
            or lock_bindings.get("fresh_outer_registry_sha256")
                != EXPECTED_REGISTRY_SHA256
            or lock_bindings.get("fresh_outer_registry_report_sha256")
                != EXPECTED_REGISTRY_REPORT_SHA256
            or lock_bindings.get("outer_hash_projection_sha256")
                != EXPECTED_PROJECTION_SHA256
            or lock_bindings.get("program_sha256") != EXPECTED_PROGRAM_SHA256
            or lock_bindings.get("selector_report_sha256")
                != EXPECTED_SELECTOR_REPORT_SHA256
            or selector.get("bindings", {}).get("program_sha256")
                != EXPECTED_PROGRAM_SHA256
            or program.get("metadata", {}).get("runtime_action_override") is not False):
        raise ValueError("v10 locked candidate chain differs")
    collection_bindings = collection.get("bindings", {})
    if (not _content_valid(collection)
            or collection_bindings.get("rows_sha256")
                != EXPECTED_COLLECTION_ROWS_SHA256
            or collection_bindings.get("candidate_lock_sha256")
                != EXPECTED_CANDIDATE_LOCK_SHA256
            or collection_bindings.get("program_sha256") != EXPECTED_PROGRAM_SHA256
            or collection.get("collection", {}).get("scored") is not False):
        raise ValueError("v10 collection chain differs")
    if (not _content_valid(outer_anchor) or not _content_valid(outer_result)
            or outer_anchor.get("attempt_key") != OUTER_ATTEMPT_KEY
            or outer_result.get("attempt_key") != OUTER_ATTEMPT_KEY
            or outer_result.get("status")
                != "passed_fresh_development_outer_gates"
            or outer_result.get("gate", {}).get("passed") is not True
            or outer_result.get("candidate_lock_sha256")
                != EXPECTED_CANDIDATE_LOCK_SHA256
            or outer_result.get("program_sha256") != EXPECTED_PROGRAM_SHA256
            or outer_result.get("outer_rows_sha256")
                != EXPECTED_COLLECTION_ROWS_SHA256):
        raise ValueError("Passed v10 outer chain differs")
    if (not _content_valid(final_anchor) or not _content_valid(final_completion)
            or final_anchor.get("attempt_key") != FINAL_ATTEMPT_KEY
            or final_anchor.get("attempt_key_inputs", {}).get("outer_attempt_key")
                != OUTER_ATTEMPT_KEY
            or final_anchor.get("attempt_key_inputs", {}).get("outer_result_sha256")
                != EXPECTED_OUTER_RESULT_SHA256
            or final_anchor.get("bindings", {}).get(
                "final_materializer_source_closure_sha256")
                != HISTORICAL_MATERIALIZER_CLOSURE_SHA256
            or final_anchor.get("bindings", {}).get(
                "final_controller_source_closure_sha256")
                != HISTORICAL_CONTROLLER_CLOSURE_SHA256
            or final_completion.get("status") != "burned_failed"
            or final_completion.get("reason") != "protected_final_phase_failed"
            or final_completion.get("final_consumed") is not True
            or final_completion.get("retry_allowed") is not False
            or final_completion.get("attempt_anchor_content_sha256")
                != final_anchor["content_sha256"]):
        raise ValueError("Burned v10 final chain differs")

    config_paths = config.get("paths")
    if (not _content_valid(config) or not isinstance(config_paths, Mapping)
            or config.get("version")
                != "warehouse-r41-diagnostic-final-materializer.v9.config.v1"
            or Path(config_paths.get("development_rows", "")).expanduser().absolute()
                != development_path
            or Path(config_paths.get("fresh_outer_hash_projection", "")).expanduser().absolute()
                != projection_path
            or type(config_paths.get("private_salt")) is not str
            or not config_paths["private_salt"]):
        raise ValueError("Burned v10 materializer configuration differs")

    historical = _historical_materializer_bytes()
    order = _call_order(historical)
    required = ["_authenticate_claim", "_load_config",
                "_authenticate_public_inputs", "_prepare_selection",
                "_read_committed_salt", "_build_material"]
    positions = [order.index(name) for name in required]
    if positions != sorted(positions):
        raise ValueError("Historical materializer salt-read order differs")
    text = historical.decode("utf-8")
    if ("Development and fresh outer isolation differs" not in text
            or "development_hashes & fresh_outer_hashes" not in text):
        raise ValueError("Historical pre-secret overlap guard differs")
    development_hashes = _project_hashes(development_path)
    overlap = set(development_hashes) & set(unique_hashes)
    overlap_rows = sum(value in overlap for value in development_hashes)
    if (len(overlap) != EXPECTED_OVERLAP_UNIQUE_HASHES
            or overlap_rows != EXPECTED_OVERLAP_ROWS):
        raise ValueError("Historical pre-secret overlap failure differs")

    bindings = {
        "v10_registry_sha256": file_hash(registry_path),
        "v10_registry_content_sha256": registry["content_sha256"],
        "v10_registry_report_sha256": file_hash(report_path),
        "v10_projection_sha256": file_hash(projection_path),
        "v10_projection_content_sha256": projection["content_sha256"],
        "v10_projection_receipt_sha256": file_hash(projection_receipt_path),
        "v10_collection_rows_sha256": file_hash(rows_path),
        "v10_collection_report_sha256": file_hash(collection_path),
        "v10_candidate_lock_sha256": file_hash(lock_path),
        "v10_selector_report_sha256": file_hash(selector_path),
        "v10_program_sha256": file_hash(program_path),
        "v10_outer_anchor_sha256": file_hash(outer_anchor_path),
        "v10_outer_result_sha256": file_hash(outer_result_path),
        "v10_final_anchor_sha256": file_hash(final_anchor_path),
        "v10_final_completion_sha256": file_hash(final_completion_path),
        "materializer_config_sha256": file_hash(config_path),
        "development_rows_sha256": file_hash(development_path),
        "historical_materializer_sha256": HISTORICAL_MATERIALIZER_SHA256,
        "contract_sha256": digest(contract()),
    }
    receipt: dict[str, Any] = {
        "version": VERSION, "status": STATUS,
        "closeout_key": digest({
            "version": VERSION,
            "final_attempt_key": FINAL_ATTEMPT_KEY,
            "final_anchor_sha256": EXPECTED_FINAL_ANCHOR_SHA256,
            "final_completion_sha256": EXPECTED_FINAL_COMPLETION_SHA256,
        }),
        "contract": contract(), "bindings": bindings,
        "burned_final": {
            "attempt_key": FINAL_ATTEMPT_KEY,
            "claim_consumed": True, "retry_allowed": False,
            "failure_stage": "public_input_authentication_before_private_salt",
            "historical_commit": HISTORICAL_COMMIT,
            "historical_materializer_source_closure_sha256": (
                HISTORICAL_MATERIALIZER_CLOSURE_SHA256),
            "historical_controller_source_closure_sha256": (
                HISTORICAL_CONTROLLER_CLOSURE_SHA256),
            "call_order": required,
            "pre_secret_overlap_unique_hashes": len(overlap),
            "pre_secret_overlap_rows": overlap_rows,
            "private_salt_path_opened": False,
            "protected_final_identity_revealed": False,
        },
        "consumed_outer": {
            "attempt_key": OUTER_ATTEMPT_KEY,
            "scene_count": SCENE_COUNT,
            "identities": public,
            "identities_sha256": digest(public),
            "family_counts": deepcopy(FAMILY_COUNTS),
            "observation_hash_projection": {
                "source_projection_sha256": EXPECTED_PROJECTION_SHA256,
                "source_projection_content_sha256": projection["content_sha256"],
                "outer_observation_hashes": unique_hashes,
                "unique_outer_observation_hash_count": len(unique_hashes),
                "outer_observation_hashes_sha256": digest(unique_hashes),
                "raw_observations_included": False,
                "actions_included": False, "probabilities_included": False,
                "labels_included": False,
            },
        },
        "disposition": {
            "same_final_attempt_retry_permitted": False,
            "v10_outer_eligible_for_development_use": True,
            "v10_outer_eligible_for_outer_claim": False,
            "v10_outer_identity_reuse_permitted": False,
            "v10_outer_observation_hash_reuse_permitted": False,
            "replacement_requires_fresh_outer_and_new_candidate_lock": True,
            "formal_ready": False,
        },
        "information_boundary": {
            "private_salt_path_treated_as_unopened_string": True,
            "private_salt_statted": False, "private_salt_read": False,
            "protected_final_identity_or_rows_read": False,
            "v10_outer_raw_observations_published": False,
            "v10_outer_actions_probabilities_or_labels_published": False,
            "formal_ready": False,
        },
        "producer_sources": sources,
        "producer_sources_sha256": digest(sources), "formal_ready": False,
    }
    receipt["content_sha256"] = digest(receipt)
    if producer_sources() != sources:
        raise RuntimeError("v11 final closeout source closure changed")
    return receipt


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    raw = (canonical(value) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                         | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def build(*, output: str | Path, permanent_closeout_registry: str | Path,
          **kwargs: Any) -> dict[str, Any]:
    receipt = create_closeout(**kwargs)
    permanent = _directory(permanent_closeout_registry,
                           "permanent v11 final-closeout registry")
    campaign = permanent / receipt["closeout_key"]
    os.mkdir(campaign, 0o700)
    _write_exclusive(campaign / RECEIPT_NAME, receipt)
    destination = Path(output).expanduser().absolute()
    parent = _directory(destination.parent, "v11 final closeout output parent")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("v11 final closeout output already exists")
    temporary = Path(tempfile.mkdtemp(
        prefix="." + destination.name + ".tmp-", dir=parent)).absolute()
    try:
        _write_exclusive(temporary / RECEIPT_NAME, receipt)
        os.rename(temporary, destination); temporary = None
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
    return deepcopy(receipt)


def read_saved_closeout(
    path: str | Path, *, expected_closeout_sha256: str,
    permanent_closeout_registry: str | Path,
) -> dict[str, Any]:
    receipt_path, value = _strict_json(
        path, "v11 burned-final closeout",
        expected_sha256=expected_closeout_sha256)
    consumed = value.get("consumed_outer")
    projection = consumed.get("observation_hash_projection") \
        if isinstance(consumed, Mapping) else None
    identities = consumed.get("identities") if isinstance(consumed, Mapping) else None
    if (value.get("version") != VERSION or value.get("status") != STATUS
            or not _content_valid(value) or value.get("contract") != contract()
            or value.get("formal_ready") is not False
            or value.get("burned_final", {}).get("private_salt_path_opened") is not False
            or value.get("burned_final", {}).get(
                "protected_final_identity_revealed") is not False
            or not isinstance(identities, list) or len(identities) != SCENE_COUNT
            or consumed.get("identities_sha256") != digest(identities)
            or not isinstance(projection, Mapping)
            or projection.get("outer_observation_hashes")
                != sorted(set(projection.get("outer_observation_hashes", [])))
            or projection.get("unique_outer_observation_hash_count") != 31150
            or projection.get("outer_observation_hashes_sha256")
                != digest(projection.get("outer_observation_hashes"))
            or value.get("disposition", {}).get(
                "v10_outer_identity_reuse_permitted") is not False
            or value.get("disposition", {}).get(
                "v10_outer_observation_hash_reuse_permitted") is not False
            or value.get("information_boundary", {}).get("private_salt_read") is not False
            or value.get("producer_sources_sha256")
                != digest(dict(sorted(value.get("producer_sources", {}).items())))):
        raise ValueError("Saved v11 final closeout semantics differ")
    permanent = _directory(permanent_closeout_registry,
                           "permanent v11 final-closeout registry")
    permanent_receipt = _regular(
        permanent / value["closeout_key"] / RECEIPT_NAME,
        "permanent v11 final closeout", maximum=MAX_JSON_BYTES,
        expected_sha256=expected_closeout_sha256)
    if permanent_receipt.read_bytes() != receipt_path.read_bytes():
        raise ValueError("Permanent v11 final closeout bytes differ")
    return deepcopy(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "v10-registry", "v10-registry-report", "v10-projection",
        "v10-projection-receipt", "v10-collection-rows",
        "v10-collection-report", "v10-collection-projection-copy",
        "v10-candidate-lock", "v10-selector-report", "v10-program",
        "v10-outer-anchor", "v10-outer-result",
        "permanent-v10-outer-registry", "v10-final-anchor",
        "v10-final-completion", "permanent-v10-final-registry",
        "materializer-config", "development-rows",
        "permanent-closeout-registry", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    values = vars(args)
    result = build(
        output=values.pop("output"),
        permanent_closeout_registry=values.pop("permanent_closeout_registry"),
        **{name.replace("_", "_") + "_path" if name not in {
            "permanent_v10_outer_registry", "permanent_v10_final_registry"
        } else name: value for name, value in values.items()})
    print(canonical({"status": result["status"],
                     "closeout_key": result["closeout_key"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VERSION", "STATUS", "RECEIPT_NAME", "FINAL_ATTEMPT_KEY",
    "OUTER_ATTEMPT_KEY", "FAMILY_COUNTS", "SCENE_COUNT", "contract",
    "producer_sources", "create_closeout", "build", "read_saved_closeout",
    "main",
]
