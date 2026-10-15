"""Strict reader for the frozen r4.1 diagnostic scene manifest.

The historical scene producer is deliberately allowed to drift after the
manifest is frozen.  This reader authenticates the saved bytes first, checks
the producer and workload bindings recorded inside those bytes, and then uses
the current strict scene/receipt validators to authenticate every saved row.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Literal, Mapping

from backend.training import warehouse_r41_diagnostic_conflict_scenarios as scenes_api
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_r41_diagnostic_workload_screen import (
    CONTRACT_SHA256 as WORKLOAD_CONTRACT_SHA256,
    FROZEN_ACTOR_SHA256,
    VERSION as WORKLOAD_SCREEN_VERSION,
    contract as workload_contract,
    load_frozen_actor,
    replay_and_compare as replay_workload_and_compare,
)
from backend.warehouse_r41_diagnostic_online_runtime import (
    R41DiagnosticOnlineAlignmentRuntime,
    diagnostic_runtime_sources,
)
from backend.training.warehouse_native_common import file_hash
from env.warehouse.domain import collaborative_study_config
from env.warehouse.navigation import ACTIONS
from env.warehouse_native.r41_conflict import canonical, digest
from env.warehouse_native.r41_diagnostic_conflict import (
    CONFLICT_FAMILIES,
    CONFLICT_FAMILIES_SHA256,
    DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
    DIAGNOSTIC_CONTRACT_SHA256,
    DIAGNOSTIC_CONTRACT_VERSION,
    diagnostic_graph_invariant_audit,
)


VERSION = "warehouse-r41-diagnostic-frozen-manifest-reader.v2"
ReplayScope = Literal["none", "development", "all"]
REPLAY_SCOPES = ("none", "development", "all")
DEVELOPMENT_REPLAY_SPLITS = (
    "train", "conflict_validation", "tutorial", "question_bank"
)
EXPECTED_MANIFEST_SHA256 = (
    "af985e9d6f041668ff1250e19da21a078ab5ccc68ecc7aca8d696f2c56845d4c"
)
EXPECTED_MANIFEST_CONTENT_SHA256 = (
    "99bf5a84f409e1411f2ee6da9b8c785aa2bec1fe6fc55161a84b412f260d57c9"
)
EXPECTED_MANIFEST_SEMANTIC_SHA256 = (
    "fa8875550bfecf2ea5fa3d47634e2f1f3c064fc8ed42db19f63af7c9fd521c2c"
)
EXPECTED_VALIDATION_SHA256 = (
    "57ea9a2c581a4015df0fd161e14089ebc4f3aa2500d34ae492bd7319f66a9fee"
)
EXPECTED_FINAL_IDENTITY_SHA256 = (
    "790cd0ee1d2a3cd4360e970e598a9b2952a390c7acd104982b4a32ef0813352d"
)
EXPECTED_FINAL_FINGERPRINTS_SHA256 = (
    "f2e9af9823954a68bc5d187dda9c5e5a67f326525ca6feb791bf9f8dd14e380d"
)
EXPECTED_FINAL_SEEDS_SHA256 = (
    "d9aa12ff763b112ae9509fa5babb640e61d1928046522ef0dd910dccd5adb136"
)
# The full scene manifest is never decoded on a development path.  These
# commitments were frozen from its development-only projection before this
# reader was introduced.  The projection is reproduced from the deterministic
# public generator and must match every commitment before any scene is handed
# to a development consumer.
EXPECTED_DEVELOPMENT_SPLITS = {
    "train": {
        "count": 128,
        "content_sha256": "e637eff2b37a65f67124becb92b2f8baa382ef4efba365650bc9cdc126e86c56",
        "identity_sha256": "bdc65cc1f97611a51826af1c384ef5ab1e484bf68ca5fe821b93386c9f052f78",
        "report_sha256": "bd0fb69b3ed7dca07d67b948fac08ad1f1af467e2d8c8b3f774a025604363647",
    },
    "conflict_validation": {
        "count": 64,
        "content_sha256": "7637eaccb71e61b408de1e08018e9b2a761567837660c7b68a23ffacba9dd1e6",
        "identity_sha256": "05f784ade6c5fbe1726cfd3d7d5c517cd02352cb2313acf7a43d536e81855891",
        "report_sha256": "c7d04ac82903b1a25e40d295442af5487ab144f44c4780bd511021d2cc9fcdd1",
    },
    "tutorial": {
        "count": 1,
        "content_sha256": "a6703e857b4d32dcb2359c085610f7f5c515d2be0da532b44c76be66cbe1d964",
        "identity_sha256": "32a4a3d4e6fb3125215cf193bcb7491af181062c64c9ff3c36e53d38263e4f7f",
        "report_sha256": "2f2224db14665fe83e7431d69b142602ca975154a7dbdb3f4ae6567159c1305c",
    },
    "question_bank": {
        "count": 36,
        "content_sha256": "e0bcfb73ed91d1af89b73c01a1218314f08dc0d509f6b5aeabd74d88f2e11bc3",
        "identity_sha256": "19b45e9a4ed800575d253d87bd34f30c1f21cf83a0acbb0ab0994878433f31fb",
        "report_sha256": "49f74c5c7b6d7f76811e0f46acc2b63c754f30f78197e5f73c424e8dda415003",
    },
}
EXPECTED_DEVELOPMENT_REPORTS_SHA256 = (
    "049e899023d254313abf0405fb7e10625922ecfeec0e85211ce3cf11674b2c24"
)
EXPECTED_CANDIDATE_BATCHES = (
    {
        "count": 720,
        "content_sha256": "611e6b273ca890ec51aeac47acd64ccfc51be40041ebeb9a3252b4d14bff8f03",
        "identity_sha256": "8f4f09de920e902fceb63ea2f8af0ce244b8105ca46a76bd29d38d2c2ce65b23",
        "report_sha256": "9cd4db66f744995c57ebdd79091af99522fce11bfe9566f307e53a46ef0a7e13",
    },
    {
        "count": 720,
        "content_sha256": "b832d2d306b029924a1096af0f3a5d007d5c826eb10a8cb14c23616e7974d6f6",
        "identity_sha256": "29112a256fbe643517dc5e58582b5d73ce68358464d3b3fda974c575ea4232d1",
        "report_sha256": "2618e041e67225ed0687315949cd42e6c68b3c2cc517349d1b2b55beb2da4c2a",
    },
    {
        "count": 720,
        "content_sha256": "0931daacd17a65cfd8335f74d5c6acb8dd603b35980ee7a5142ecbeb35fe633a",
        "identity_sha256": "c442105c54b8d4c829858573965d182fecd79ab0c893ebe5eb641b84abd8edd7",
        "report_sha256": "f639565b1b2a93ad6d4592144581461a2c930ab8e382865114fa1db401527a85",
    },
)
EXPECTED_CANDIDATE_BATCHES_SHA256 = (
    "41240d79260dbe76104f07a24fdcbf12a67dfd110a06909360dfd5e04dd256e5"
)
EXPECTED_CANDIDATE_REPORTS_SHA256 = (
    "3781e06d2a5371ad7ac71ca63e7ab23afb0c87d88beaeed83f1b23d32ee470ba"
)
EXPECTED_DEVELOPMENT_PROJECTION_SHA256 = (
    "02c2c3392e8d7938d6642f5d3c1fe0142dc7f5abe688df0b6a3b2e9d7aae4610"
)
EXPECTED_ACTOR_PARAMETERS_SHA256 = (
    "fc9095d0c0e230f4edf0004d66be42e0d176be166d0b071a0576d9954c057ea3"
)
EXPECTED_WORKLOAD_SOURCE_SHA256 = (
    "009bc4b999896d58e85347cb8484461e5f6cb3a45aeb94df548d80a7e8572b11"
)
EXPECTED_PRODUCER_SOURCES_SHA256 = (
    "47968ebd2b54f3305a97504e496b4b2cd7457f186c201d2550c0ab08f3ba843e"
)
EXPECTED_RUNTIME_SOURCES_SHA256 = (
    "409be07d85a11646f2b883a5ed77fead4e0ce48a6ac952827883b15223da26d8"
)
EXPECTED_SOURCE_MANIFEST = {
    "version": "warehouse-r41-conflict-scene-manifest.v1",
    "file_sha256": "4e640e19ea37405c3613d8f6ea1ace036ecff5da50cfb75c7e3f07d825f5b98e",
    "content_sha256": "6afa2efe7c7dde8e4d7297a4e5855348948817b196acbf64159867f8d3fe5c12",
}
EXPECTED_PRODUCER_SOURCE_KEYS = {
    "backend/training/warehouse_r41_diagnostic_conflict_scenarios.py",
    "backend/training/warehouse_r41_diagnostic_workload_screen.py",
    "backend/warehouse_r41_diagnostic_online_runtime.py",
    "env/warehouse_native/r41_conflict.py",
    "env/warehouse_native/r41_diagnostic_conflict.py",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TOP_LEVEL_KEYS = {
    "version",
    "configuration",
    "diagnostic_contract_version",
    "diagnostic_contract_sha256",
    "diagnostic_conflict_graph_sha256",
    "diagnostic_graph_invariant_audit",
    "diagnostic_graph_invariant_audit_sha256",
    "conflict_families_sha256",
    "conflict_families",
    "frozen_actor",
    "workload_screen",
    "producer_sources",
    "producer_sources_sha256",
    "source_r41_manifest",
    "protocol",
    "splits",
    "workload_generation_reports",
    "candidate_batches",
    "batch_reports",
    "content_sha256",
}
_DEVELOPMENT_PROJECTION_CACHE: tuple[
    tuple[str, str, str, str, str], dict[str, Any]
] | None = None


def _regular(path: str | Path, label: str) -> Path:
    value = Path(path).expanduser().absolute()
    if value.is_symlink() or not value.is_file() or value.resolve() != value:
        raise ValueError(label + " must be a canonical regular file")
    return value


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Frozen diagnostic manifest contains a duplicate key")
        result[key] = value
    return result


def _reject_nonfinite(value: str):
    raise ValueError("Frozen diagnostic manifest contains a non-finite number: " + value)


def _read_json_object(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Frozen diagnostic manifest is not strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError("Frozen diagnostic manifest must be a JSON object")
    return value


def _expected_protocol() -> dict[str, Any]:
    return {
        "per_family_per_batch": scenes_api.PER_FAMILY_PER_BATCH,
        "batch_count": scenes_api.MAXIMUM_BATCHES,
        "maximum_batches": scenes_api.MAXIMUM_BATCHES,
        "candidate_public_start_positions": "deterministic clear passable cells",
        "candidate_initial_battery": [100.0, 100.0],
        "candidate_successor_stream": (
            "unique deterministic stream bound to scene seed, batch, and family"
        ),
        "successor_graph": "diagnostic v2 robust fixed-point closure",
        "successor_endpoint_occupancy": (
            "pickup and delivery both clear of both post-motion robots"
        ),
        "immediate_task_recreation": False,
        "ordinary_task_fallback": False,
        "participant_data_read": False,
        "final_test_used_for_selection": False,
        "base_scene_policy": (
            "fresh deterministic draws admitted only after exact split workload replay"
        ),
        "base_split_seed_starts": deepcopy(scenes_api.SPLIT_SEED_STARTS),
        "workload_unsafe_scene_policy": (
            "reject and advance seed; strict sampler remains fail closed"
        ),
    }


def runtime_sources() -> dict[str, Any]:
    """Authenticate and return the complete current physical-runtime closure."""
    sources = dict(sorted(diagnostic_runtime_sources().items()))
    closure_sha256 = digest(sources)
    if closure_sha256 != EXPECTED_RUNTIME_SOURCES_SHA256:
        raise ValueError("Diagnostic runtime source closure differs from the v2 binding")
    return {
        "version": VERSION,
        "sources": deepcopy(sources),
        "sources_sha256": closure_sha256,
    }


def manifest_identity() -> dict[str, str]:
    """Return fixed full-manifest bindings without exposing final state."""
    return {
        "manifest_file_sha256": EXPECTED_MANIFEST_SHA256,
        "manifest_content_sha256": EXPECTED_MANIFEST_CONTENT_SHA256,
        "manifest_semantic_sha256": EXPECTED_MANIFEST_SEMANTIC_SHA256,
        "validation_file_sha256": EXPECTED_VALIDATION_SHA256,
        "protected_final_identity_sha256": EXPECTED_FINAL_IDENTITY_SHA256,
        "development_projection_sha256": EXPECTED_DEVELOPMENT_PROJECTION_SHA256,
    }


def _development_projection_key(
    actor_file: Path,
) -> tuple[str, str, str, str, str]:
    workload_path = Path(replay_workload_and_compare.__code__.co_filename).resolve()
    workload_sha256 = file_hash(workload_path)
    if workload_sha256 != EXPECTED_WORKLOAD_SOURCE_SHA256:
        raise ValueError("Diagnostic workload source differs from frozen manifest")
    return (
        file_hash(actor_file),
        file_hash(Path(scenes_api.__file__).resolve()),
        workload_sha256,
        runtime_sources()["sources_sha256"],
        # The deterministic generator calls into a wider set of local
        # evaluation, partner, environment, and runtime modules than the two
        # direct files above.  Bind the cache to that complete executable
        # closure so a long-lived process can never reuse a projection after
        # any transitive producer source changes.
        digest(local_source_hashes((Path(__file__).resolve(),))),
    )


def _development_projection(actor_file: Path) -> dict[str, Any]:
    """Regenerate once per stable process and return a detached projection."""
    global _DEVELOPMENT_PROJECTION_CACHE
    before = _development_projection_key(actor_file)
    if (_DEVELOPMENT_PROJECTION_CACHE is not None
            and _DEVELOPMENT_PROJECTION_CACHE[0] == before):
        return deepcopy(_DEVELOPMENT_PROJECTION_CACHE[1])
    splits, reports = regenerate_development_splits(
        actor_file, splits=DEVELOPMENT_REPLAY_SPLITS)
    batches, batch_reports = regenerate_development_candidate_batches()
    projection = {
        "splits": splits,
        "workload_generation_reports": reports,
        "candidate_batches": batches,
        "batch_reports": batch_reports,
    }
    if digest(projection) != EXPECTED_DEVELOPMENT_PROJECTION_SHA256:
        raise ValueError("Regenerated development projection differs")
    if _development_projection_key(actor_file) != before:
        raise RuntimeError("Development projection source changed during replay")
    _DEVELOPMENT_PROJECTION_CACHE = (before, deepcopy(projection))
    return deepcopy(projection)


def build_runtime(
    *, actor_path: str | Path, protocol_path: str | Path,
    manifest_path: str | Path,
) -> R41DiagnosticOnlineAlignmentRuntime:
    """Build the physical runtime from fixed full-manifest commitments."""
    actor_file = _regular(actor_path, "Frozen diagnostic Actor")
    protocol_file = _regular(protocol_path, "Frozen diagnostic protocol")
    manifest_file = Path(manifest_path).expanduser().absolute()
    if (manifest_file.is_symlink() or not manifest_file.is_file()
            or manifest_file.resolve() != manifest_file):
        raise ValueError("Frozen diagnostic manifest must be a canonical regular file")
    if file_hash(actor_file) != FROZEN_ACTOR_SHA256:
        raise ValueError("Frozen diagnostic Actor bytes differ")
    if file_hash(manifest_file) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Frozen diagnostic manifest bytes differ")
    protocol = _read_json_object(protocol_file.read_bytes())
    portable = {
        "version": "warehouse-r41-diagnostic-portable-runtime-manifest.v1",
        "source_full_manifest_version": scenes_api.VERSION,
        "source_conflict_validation_version": scenes_api.VALIDATION_VERSION,
        "source_conflict_validation_sha256": EXPECTED_VALIDATION_SHA256,
        "source_full_manifest_file_sha256": EXPECTED_MANIFEST_SHA256,
        "source_full_manifest_content_sha256": EXPECTED_MANIFEST_CONTENT_SHA256,
        "source_full_manifest_semantic_sha256": EXPECTED_MANIFEST_SEMANTIC_SHA256,
        "diagnostic_contract_version": DIAGNOSTIC_CONTRACT_VERSION,
        "diagnostic_contract_sha256": DIAGNOSTIC_CONTRACT_SHA256,
        "diagnostic_conflict_graph_sha256": DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
        "conflict_families_sha256": CONFLICT_FAMILIES_SHA256,
    }
    portable["content_sha256"] = digest(portable)
    temporary = tempfile.TemporaryDirectory(
        prefix="warehouse-r41-development-runtime-")
    portable_path = Path(temporary.name).resolve() / "portable_manifest.json"
    descriptor = os.open(
        portable_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write((canonical(portable) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    runtime = R41DiagnosticOnlineAlignmentRuntime(
        actor_file,
        training_protocol_path=protocol_file,
        manifest_path=portable_path,
        expected_actor_sha256=FROZEN_ACTOR_SHA256,
        expected_training_protocol_file_sha256=file_hash(protocol_file),
        expected_training_protocol_content_sha256=digest(protocol),
        expected_manifest_file_sha256=file_hash(portable_path),
        expected_manifest_content_sha256=portable["content_sha256"],
        expected_manifest_semantic_sha256=digest(portable),
    )
    # Keep the private portable manifest alive for runtime.verify_binding().
    runtime._development_manifest_temporary = temporary
    return runtime


def _validate_source_bindings(manifest: Mapping[str, Any]) -> None:
    actor = manifest.get("frozen_actor")
    workload = manifest.get("workload_screen")
    producer_sources = manifest.get("producer_sources")
    source_manifest = manifest.get("source_r41_manifest")
    if (
        not isinstance(actor, Mapping)
        or set(actor) != {"sha256", "actor_parameters_sha256"}
        or actor.get("sha256") != FROZEN_ACTOR_SHA256
        or actor.get("actor_parameters_sha256")
        != EXPECTED_ACTOR_PARAMETERS_SHA256
    ):
        raise ValueError("Frozen diagnostic Actor binding differs")
    if (
        not isinstance(workload, Mapping)
        or set(workload) != {"version", "contract", "contract_sha256", "source_sha256"}
        or workload.get("version") != WORKLOAD_SCREEN_VERSION
        or canonical(workload.get("contract")) != canonical(workload_contract())
        or workload.get("contract_sha256") != WORKLOAD_CONTRACT_SHA256
        or workload.get("contract_sha256") != digest(workload.get("contract"))
        or workload.get("source_sha256") != EXPECTED_WORKLOAD_SOURCE_SHA256
    ):
        raise ValueError("Frozen diagnostic workload binding differs")
    if (
        not isinstance(producer_sources, Mapping)
        or set(producer_sources) != EXPECTED_PRODUCER_SOURCE_KEYS
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in producer_sources.values()
        )
        or producer_sources.get(
            "backend/training/warehouse_r41_diagnostic_workload_screen.py"
        )
        != workload.get("source_sha256")
        or manifest.get("producer_sources_sha256") != digest(producer_sources)
        or manifest.get("producer_sources_sha256")
        != EXPECTED_PRODUCER_SOURCES_SHA256
    ):
        raise ValueError("Frozen diagnostic producer source binding differs")
    if (
        not isinstance(source_manifest, Mapping)
        or set(source_manifest) != {"path", *EXPECTED_SOURCE_MANIFEST}
        or not isinstance(source_manifest.get("path"), str)
        or any(
            source_manifest.get(key) != value
            for key, value in EXPECTED_SOURCE_MANIFEST.items()
        )
    ):
        raise ValueError("Frozen diagnostic source-manifest binding differs")


def _validate_contract(manifest: Mapping[str, Any]) -> None:
    graph_audit = diagnostic_graph_invariant_audit()
    if set(manifest) != _TOP_LEVEL_KEYS:
        raise ValueError("Frozen diagnostic manifest schema differs")
    if (
        manifest.get("version") != scenes_api.VERSION
        or canonical(manifest.get("configuration"))
        != canonical(asdict(collaborative_study_config()))
        or manifest.get("diagnostic_contract_version")
        != DIAGNOSTIC_CONTRACT_VERSION
        or manifest.get("diagnostic_contract_sha256") != DIAGNOSTIC_CONTRACT_SHA256
        or manifest.get("diagnostic_conflict_graph_sha256")
        != DIAGNOSTIC_CONFLICT_GRAPH_SHA256
        or canonical(manifest.get("diagnostic_graph_invariant_audit"))
        != canonical(graph_audit)
        or manifest.get("diagnostic_graph_invariant_audit_sha256")
        != digest(graph_audit)
        or graph_audit.get("passed") is not True
        or manifest.get("conflict_families_sha256") != CONFLICT_FAMILIES_SHA256
        or canonical(manifest.get("conflict_families"))
        != canonical(CONFLICT_FAMILIES)
        or canonical(manifest.get("protocol")) != canonical(_expected_protocol())
    ):
        raise ValueError("Frozen diagnostic manifest contract differs")


def _balanced_family_counts(count: int) -> dict[str, int]:
    quotient, remainder = divmod(count, len(scenes_api.FAMILY_IDS))
    return {
        family: quotient + int(index < remainder)
        for index, family in enumerate(scenes_api.FAMILY_IDS)
    }


def _validate_generation_report(
    report: Mapping[str, Any],
    *,
    split: str,
    rows: list[Mapping[str, Any]],
) -> None:
    counts = dict(sorted(Counter(row["family_id"] for row in rows).items()))
    expected_counts = (
        counts
        if split == "tutorial"
        else _balanced_family_counts(scenes_api.EXPECTED_BASE_COUNTS[split])
    )
    draws = report.get("draws")
    rejected = report.get("rejected_workload_unsafe")
    rejected_by_family = report.get("rejected_by_family")
    failure_types = report.get("failure_types")
    if (
        report.get("split") != split
        or report.get("seed_start") != scenes_api.SPLIT_SEED_STARTS[split]
        or type(draws) is not int
        or draws < len(rows)
        or report.get("last_seed_examined")
        != scenes_api.SPLIT_SEED_STARTS[split] + draws - 1
        or report.get("accepted") != len(rows)
        or report.get("accepted_by_family") != expected_counts
        or type(rejected) is not int
        or rejected < 0
        or not isinstance(rejected_by_family, Mapping)
        or any(type(value) is not int or value < 0 for value in rejected_by_family.values())
        or sum(rejected_by_family.values()) != rejected
        or not isinstance(failure_types, Mapping)
        or any(type(value) is not int or value < 0 for value in failure_types.values())
        or sum(failure_types.values()) != rejected
        or not isinstance(report.get("failure_examples"), list)
        or len(report["failure_examples"]) > min(12, rejected)
        or report.get("accepted_receipts_sha256")
        != digest([row["workload_screen"]["receipt_sha256"] for row in rows])
    ):
        raise ValueError("Frozen diagnostic workload generation report differs")


def _validate_rows(
    manifest: Mapping[str, Any],
    *,
    workload_actor,
    replay_scope: ReplayScope,
) -> None:
    splits = manifest.get("splits")
    reports = manifest.get("workload_generation_reports")
    if (
        not isinstance(splits, Mapping)
        or set(splits) != set(scenes_api.BASE_SPLITS)
        or not isinstance(reports, list)
        or len(reports) != len(scenes_api.BASE_SPLITS)
        or [row.get("split") if isinstance(row, Mapping) else None for row in reports]
        != list(scenes_api.BASE_SPLITS)
    ):
        raise ValueError("Frozen diagnostic base-split schema differs")

    seen_ids: set[str] = set()
    seen_seeds: set[int] = set()
    seen_fingerprints: set[str] = set()
    for split, report in zip(scenes_api.BASE_SPLITS, reports):
        rows = splits[split]
        if (
            not isinstance(rows, list)
            or len(rows) != scenes_api.EXPECTED_BASE_COUNTS[split]
            or not isinstance(report, Mapping)
        ):
            raise ValueError("Frozen diagnostic base-split count differs")
        previous_seed = -1
        for scene_index, scene in enumerate(rows):
            if not isinstance(scene, Mapping):
                raise ValueError("Frozen diagnostic scene must be an object")
            if split == "final_test" and replay_scope != "all":
                identity = {
                    "id": scene.get("id"),
                    "seed": scene.get("seed"),
                    "fingerprint": scene.get("fingerprint"),
                }
                if (
                    identity["id"] != f"diagnostic_final_test_{scene_index:04d}"
                    or type(identity["seed"]) is not int
                    or identity["seed"] <= previous_seed
                    or not isinstance(identity["fingerprint"], str)
                    or _SHA256.fullmatch(identity["fingerprint"]) is None
                ):
                    raise ValueError("Frozen diagnostic final identity differs")
                previous_seed = identity["seed"]
                if (
                    identity["id"] in seen_ids
                    or identity["seed"] in seen_seeds
                    or identity["fingerprint"] in seen_fingerprints
                ):
                    raise ValueError("Frozen diagnostic scene splits overlap")
                seen_ids.add(identity["id"])
                seen_seeds.add(identity["seed"])
                seen_fingerprints.add(identity["fingerprint"])
                continue
            scenes_api._validate_scene(
                scene,
                expected_split=split,
                expected_batch=None,
                expected_index=scene_index,
            )
            if scene.get("id") != f"diagnostic_{split}_{scene_index:04d}":
                raise ValueError("Frozen diagnostic base-scene id differs")
            if scene.get("seed", -1) <= previous_seed:
                raise ValueError("Frozen diagnostic base-scene seeds are not ordered")
            previous_seed = scene["seed"]
            expected_stream = scenes_api._successor_stream_seed(
                scene["seed"], -1, scene["family_id"]
            )
            if scene.get("successor_stream_seed") != expected_stream:
                raise ValueError("Frozen diagnostic successor stream differs")
            if (
                replay_scope == "all"
                or (
                    replay_scope == "development"
                    and split in DEVELOPMENT_REPLAY_SPLITS
                )
            ):
                replay_workload_and_compare(
                    scene,
                    split=split,
                    scene_index=scene_index,
                    actor=(None if split == "tutorial" else workload_actor),
                    train_count=scenes_api.EXPECTED_BASE_COUNTS["train"],
                )
            identity = (scene["id"], scene["seed"], scene["fingerprint"])
            if (
                identity[0] in seen_ids
                or identity[1] in seen_seeds
                or identity[2] in seen_fingerprints
            ):
                raise ValueError("Frozen diagnostic scene splits overlap")
            seen_ids.add(identity[0])
            seen_seeds.add(identity[1])
            seen_fingerprints.add(identity[2])
        if split != "final_test" or replay_scope == "all":
            _validate_generation_report(report, split=split, rows=rows)

    final_identities = [
        {"id": row.get("id"), "seed": row.get("seed"),
         "fingerprint": row.get("fingerprint")}
        for row in splits["final_test"]
    ]
    if (
        digest(final_identities) != EXPECTED_FINAL_IDENTITY_SHA256
        or digest(sorted(row["fingerprint"] for row in final_identities))
        != EXPECTED_FINAL_FINGERPRINTS_SHA256
        or digest(sorted(row["seed"] for row in final_identities))
        != EXPECTED_FINAL_SEEDS_SHA256
    ):
        raise ValueError("Frozen diagnostic final identity registry differs")

    batches = manifest.get("candidate_batches")
    batch_reports = manifest.get("batch_reports")
    if (
        not isinstance(batches, list)
        or not isinstance(batch_reports, list)
        or len(batches) != scenes_api.MAXIMUM_BATCHES
        or len(batch_reports) != scenes_api.MAXIMUM_BATCHES
    ):
        raise ValueError("Frozen diagnostic candidate-batch schema differs")
    per_family = scenes_api.PER_FAMILY_PER_BATCH
    for batch_index, (rows, report) in enumerate(zip(batches, batch_reports)):
        if not isinstance(rows, list) or not isinstance(report, Mapping):
            raise ValueError("Frozen diagnostic candidate batch differs")
        counts = Counter(row.get("family_id") for row in rows if isinstance(row, Mapping))
        expected_counts = Counter(
            {family: per_family for family in scenes_api.FAMILY_IDS}
        )
        if len(rows) != per_family * len(scenes_api.FAMILY_IDS) or counts != expected_counts:
            raise ValueError("Frozen diagnostic candidate family quota differs")
        family_offsets = Counter()
        previous_order = None
        for scene in rows:
            if not isinstance(scene, Mapping):
                raise ValueError("Frozen diagnostic candidate scene must be an object")
            family = scene["family_id"]
            offset = family_offsets[family]
            family_offsets[family] += 1
            scenes_api._validate_scene(
                scene,
                expected_split="play_candidates",
                expected_batch=batch_index,
            )
            if scene.get("id") != f"diagnostic_b{batch_index}_{family}_{offset:03d}":
                raise ValueError("Frozen diagnostic candidate id differs")
            order = (scenes_api.FAMILY_IDS.index(family), scene["seed"])
            if previous_order is not None and order <= previous_order:
                raise ValueError("Frozen diagnostic candidate order differs")
            previous_order = order
            expected_stream = scenes_api._successor_stream_seed(
                scene["seed"], batch_index, family
            )
            if scene.get("successor_stream_seed") != expected_stream:
                raise ValueError("Frozen diagnostic successor stream differs")
            identity = (scene["id"], scene["seed"], scene["fingerprint"])
            if (
                identity[0] in seen_ids
                or identity[1] in seen_seeds
                or identity[2] in seen_fingerprints
            ):
                raise ValueError("Frozen diagnostic candidate pools overlap")
            seen_ids.add(identity[0])
            seen_seeds.add(identity[1])
            seen_fingerprints.add(identity[2])
        draws = report.get("draws")
        if (
            report.get("batch_index") != batch_index
            or report.get("seed_start") != scenes_api.BATCH_SEED_STARTS[batch_index]
            or type(draws) is not int
            or draws < len(rows)
            or report.get("candidate_count") != len(rows)
            or report.get("per_family") != dict(sorted(expected_counts.items()))
            or report.get("candidate_fingerprints_sha256")
            != digest([row["fingerprint"] for row in rows])
        ):
            raise ValueError("Frozen diagnostic candidate generation report differs")

    question_metrics = [
        scene["workload_screen"]["metrics"] for scene in splits["question_bank"]
    ]
    next_actions = {
        action for metrics in question_metrics for action in metrics["next_actions"]
    }
    wait_displacements = {
        tuple(value)
        for metrics in question_metrics
        for value in metrics["wait_displacements"]
    }
    if (
        len(next_actions) < 4
        or len(wait_displacements) < 4
        or any(action not in ACTIONS for action in next_actions)
    ):
        raise ValueError("Frozen diagnostic question split lacks outcome coverage")


def read_saved_manifest(
    path: str | Path,
    *,
    expected_sha256: str = EXPECTED_MANIFEST_SHA256,
    actor_path: str | Path | None = None,
    replay_scope: ReplayScope = "none",
) -> dict[str, Any]:
    """Authenticate the manifest for development without parsing scene JSON."""
    if expected_sha256 != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Frozen diagnostic manifest hash cannot be overridden")
    if replay_scope not in REPLAY_SCOPES:
        raise ValueError("Frozen diagnostic manifest replay scope is invalid")
    if replay_scope == "all":
        raise ValueError(
            "Full manifest replay requires post-success publication authorization")
    manifest_path = _regular(path, "Frozen diagnostic manifest")
    if file_hash(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Frozen diagnostic manifest bytes differ")
    validation_path = _regular(
        manifest_path.parent / "validation.json",
        "Frozen diagnostic manifest validation")
    if file_hash(validation_path) != EXPECTED_VALIDATION_SHA256:
        raise ValueError("Frozen diagnostic manifest validation bytes differ")
    runtime_sources()
    if replay_scope == "development" and actor_path is None:
        raise ValueError("Exact Actor path is required for development authentication")
    if actor_path is not None:
        actor_file = _regular(actor_path, "Frozen diagnostic Actor")
        workload_actor = load_frozen_actor(actor_file)
        if (
            workload_actor.artifact_sha256 != FROZEN_ACTOR_SHA256
            or
            workload_actor.metadata.get("actor_parameters_sha256")
            != EXPECTED_ACTOR_PARAMETERS_SHA256
        ):
            raise ValueError("Frozen diagnostic Actor parameters differ")
    result: dict[str, Any] = {
        "version": scenes_api.VERSION,
        "content_sha256": EXPECTED_MANIFEST_CONTENT_SHA256,
        "semantic_sha256": EXPECTED_MANIFEST_SEMANTIC_SHA256,
        "frozen_actor": {
            "sha256": FROZEN_ACTOR_SHA256,
            "actor_parameters_sha256": EXPECTED_ACTOR_PARAMETERS_SHA256,
        },
        "authentication": {
            "reader_version": VERSION,
            "replay_scope": replay_scope,
            "manifest_file_sha256": EXPECTED_MANIFEST_SHA256,
            "manifest_content_sha256": EXPECTED_MANIFEST_CONTENT_SHA256,
            "manifest_semantic_sha256": EXPECTED_MANIFEST_SEMANTIC_SHA256,
            "validation_file_sha256": EXPECTED_VALIDATION_SHA256,
            "full_manifest_global_scene_identity_uniqueness_authenticated": True,
            "candidate_scenes_disjoint_from_all_base_splits_authenticated": True,
            "protected_final_identity_sha256": EXPECTED_FINAL_IDENTITY_SHA256,
            "protected_final_fingerprints_sha256": (
                EXPECTED_FINAL_FINGERPRINTS_SHA256),
            "protected_final_seeds_sha256": EXPECTED_FINAL_SEEDS_SHA256,
            "full_manifest_json_parsed": False,
            "final_identity_exclusion_only": True,
            "final_scene_geometry_access": False,
            "final_trajectories_access": False,
            "final_actor_outputs_or_labels_access": False,
            "final_used_for_fit_selection_metrics": False,
        },
    }
    if replay_scope == "development":
        projection = _development_projection(actor_file)
        result.update(deepcopy(projection))
        result["authentication"].update({
            "development_projection_regenerated": True,
            "development_projection_sha256": digest(projection),
            "final_test_rows_present": False,
        })
    return result


def development_scene_splits(
    manifest: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Return only the authenticated development train/validation registries.

    The fixed manifest's final-test rows are used only as protected identities;
    no final workload metric or label is returned by this helper.
    """
    if not isinstance(manifest, Mapping):
        raise ValueError("Authenticated frozen manifest must be an object")
    if manifest.get("content_sha256") != EXPECTED_MANIFEST_CONTENT_SHA256:
        raise ValueError("Authenticated frozen manifest content differs")
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("Authenticated frozen manifest splits differ")
    result = {
        name: deepcopy(splits[name])
        for name in ("train", "conflict_validation")
    }
    train = [*result["train"], *result["conflict_validation"]]
    if len(result["train"]) != 128 or len(result["conflict_validation"]) != 64:
        raise ValueError("Authenticated development scene count differs")
    fingerprints = [row.get("fingerprint") for row in train]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError("Authenticated development scene identities overlap")
    return result


def regenerate_development_splits(
    actor_path: str | Path, *, splits: tuple[str, ...] = DEVELOPMENT_REPLAY_SPLITS,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Regenerate only public development splits without opening the manifest."""
    if (
        not splits
        or len(set(splits)) != len(splits)
        or any(name not in DEVELOPMENT_REPLAY_SPLITS for name in splits)
    ):
        raise ValueError("Only fixed development splits may be regenerated")
    actor_file = _regular(actor_path, "Frozen diagnostic Actor")
    actor = load_frozen_actor(actor_file)
    if (
        actor.artifact_sha256 != FROZEN_ACTOR_SHA256
        or actor.metadata.get("actor_parameters_sha256")
            != EXPECTED_ACTOR_PARAMETERS_SHA256
    ):
        raise ValueError("Frozen diagnostic Actor differs")
    rows: dict[str, list[dict[str, Any]]] = {}
    reports: list[dict[str, Any]] = []
    for split in splits:
        split_rows, report = scenes_api._generate_screened_split(
            split,
            seed_start=scenes_api.SPLIT_SEED_STARTS[split],
            count=scenes_api.EXPECTED_BASE_COUNTS[split],
            actor=actor,
        )
        rows[split] = split_rows
        reports.append(report)
        expected = EXPECTED_DEVELOPMENT_SPLITS[split]
        identities = [
            {"id": row.get("id"), "seed": row.get("seed"),
             "fingerprint": row.get("fingerprint")}
            for row in split_rows
        ]
        if (
            len(split_rows) != expected["count"]
            or digest(split_rows) != expected["content_sha256"]
            or digest(identities) != expected["identity_sha256"]
            or digest(report) != expected["report_sha256"]
        ):
            raise ValueError(
                "Regenerated development split differs: " + split)
    if tuple(splits) == DEVELOPMENT_REPLAY_SPLITS and (
        digest(reports) != EXPECTED_DEVELOPMENT_REPORTS_SHA256
    ):
        raise ValueError("Regenerated development reports differ")
    return rows, reports


def regenerate_development_candidate_batches(
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    """Regenerate the deterministic development candidate pool only."""
    batches: list[list[dict[str, Any]]] = []
    reports: list[dict[str, Any]] = []
    for batch_index in range(scenes_api.MAXIMUM_BATCHES):
        rows, report = scenes_api.generate_candidate_batch(
            batch_index, per_family=scenes_api.PER_FAMILY_PER_BATCH)
        expected = EXPECTED_CANDIDATE_BATCHES[batch_index]
        identities = [
            {"id": row.get("id"), "seed": row.get("seed"),
             "fingerprint": row.get("fingerprint")}
            for row in rows
        ]
        if (
            len(rows) != expected["count"]
            or digest(rows) != expected["content_sha256"]
            or digest(identities) != expected["identity_sha256"]
            or digest(report) != expected["report_sha256"]
        ):
            raise ValueError(
                "Regenerated development candidate batch differs")
        batches.append(rows)
        reports.append(report)
    if (
        digest(batches) != EXPECTED_CANDIDATE_BATCHES_SHA256
        or digest(reports) != EXPECTED_CANDIDATE_REPORTS_SHA256
    ):
        raise ValueError("Regenerated development candidate projection differs")
    return batches, reports


__all__ = [
    "VERSION",
    "EXPECTED_MANIFEST_SHA256",
    "EXPECTED_MANIFEST_CONTENT_SHA256",
    "EXPECTED_MANIFEST_SEMANTIC_SHA256",
    "EXPECTED_VALIDATION_SHA256",
    "EXPECTED_FINAL_IDENTITY_SHA256",
    "EXPECTED_FINAL_FINGERPRINTS_SHA256",
    "EXPECTED_FINAL_SEEDS_SHA256",
    "EXPECTED_RUNTIME_SOURCES_SHA256",
    "EXPECTED_DEVELOPMENT_SPLITS",
    "EXPECTED_DEVELOPMENT_REPORTS_SHA256",
    "EXPECTED_CANDIDATE_BATCHES",
    "EXPECTED_CANDIDATE_BATCHES_SHA256",
    "EXPECTED_CANDIDATE_REPORTS_SHA256",
    "EXPECTED_DEVELOPMENT_PROJECTION_SHA256",
    "REPLAY_SCOPES",
    "DEVELOPMENT_REPLAY_SPLITS",
    "runtime_sources",
    "manifest_identity",
    "build_runtime",
    "read_saved_manifest",
    "development_scene_splits",
    "regenerate_development_splits",
    "regenerate_development_candidate_batches",
]
