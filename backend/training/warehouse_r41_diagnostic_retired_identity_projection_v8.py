"""Declassify only retired v1/v2 exposed identities for development use.

The two source holdouts are permanently retired and pinned by exact file SHA.
This one-time producer is the only development utility that opens their full
JSON files.  It closes over every candidate fingerprint that entered either
selection trace, maps each fingerprint to one seed through the authenticated
public candidate pool, and unions the 64 accepted scene identities.  Its
output contains only version/source identity, the aggregate accepted count,
and exposed ``seed``/``fingerprint`` pairs.  It never copies snapshots,
workload receipts, trace rows/outcomes, statistics, metrics, or Actor outputs.

Development producers consume the strict saved projection reader below and
must never receive the full retired holdouts themselves.
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
import stat
import tempfile
from typing import Any, Callable, Mapping, Sequence

from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.training import warehouse_r41_diagnostic_frozen_manifest_v2 as manifest_binding
from backend.training.warehouse_r41_diagnostic_input_snapshot_v8 import (
    ImmutableInputSnapshot,
)
from backend.warehouse_r41_diagnostic_online_runtime import (
    R41DiagnosticConflictWarehouseEnv,
)
from env.warehouse.domain import AgentState, WarehouseState
from env.warehouse.navigation import MOVE_DELTAS
from env.warehouse_native.r41_conflict import task_node_id
from env.warehouse_native.r41_diagnostic_conflict import (
    conflict_family_id,
    diagnostic_scene_fingerprint,
    validate_diagnostic_active_conflict,
)


VERSION = "warehouse-r41-diagnostic-retired-identity-projection.v2"
REPORT_VERSION = "warehouse-r41-diagnostic-retired-identity-projection-audit.v2"
STATUS = "passed_exact_exposure_identity_only_declassification"
ROOT = Path(__file__).resolve().parents[2]
MAX_JSON_BYTES = 512 * 1024 * 1024
ACCEPTED_IDENTITIES_PER_SOURCE = 64
EXPECTED_EXPOSED_IDENTITIES_PER_SOURCE = {
    "warehouse-r41-diagnostic-fresh-final-holdout.v1": 68,
    "warehouse-r41-diagnostic-fresh-final-holdout.v2": 76,
}
EXPECTED_GLOBAL_EXPOSED_IDENTITIES = 139
EXPECTED_ACTOR_SHA256 = manifest_binding.FROZEN_ACTOR_SHA256
EXPECTED_MANIFEST_SHA256 = manifest_binding.EXPECTED_MANIFEST_SHA256
EXPECTED_MANIFEST_VALIDATION_SHA256 = manifest_binding.EXPECTED_VALIDATION_SHA256
SUPERSEDES_PROJECTION_SHA256 = (
    "02ec8c091e61668e3bbc3fddde0c2092e30e7bec322f10848851d24574ae5b83"
)
SOURCE_FILE_SHA256 = {
    "warehouse-r41-diagnostic-fresh-final-holdout.v1": (
        "6c3ff25f9917451815979173d98881fe5e5e98258b514d199d50ee362d2746c2"
    ),
    "warehouse-r41-diagnostic-fresh-final-holdout.v2": (
        "7d72424744eea7547516cffe414c15578802c7204b90bdf1b129a1e9acfbf17b"
    ),
}
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_PROJECTION_FIELDS = frozenset((
    "version", "sources", "exposed_identities", "content_sha256",
))
_SOURCE_FIELDS = frozenset((
    "version", "source_file_sha256", "accepted_count", "exposed_count",
))
_IDENTITY_FIELDS = frozenset(("seed", "fingerprint"))
_REPORT_FIELDS = frozenset((
    "version", "status", "projection_file_sha256",
    "projection_content_sha256", "source_file_sha256", "identity_counts",
    "global_identity_count", "identity_uniqueness", "declassification",
    "actor_file_sha256", "manifest_file_sha256",
    "manifest_validation_file_sha256", "development_projection_sha256",
    "supersedes_projection_sha256", "producer_sources",
    "producer_sources_sha256", "content_sha256",
))
_DECLASSIFICATION = {
    "fields_released": ["seed", "fingerprint"],
    "aggregate_accepted_count_released": True,
    "per_identity_accepted_flag_released": False,
    "full_retired_json_opened_by_one_time_builder": True,
    "development_consumers_open_full_retired_json": False,
    "source_exact_bytes_authenticated": True,
    "source_selection_trace_fingerprint_used": True,
    "source_selection_trace_other_fields_used": False,
    "source_selection_trace_rows_released": False,
    "source_statistics_or_metrics_used": False,
    "source_snapshot_or_workload_used": False,
    "public_candidate_batches_used_only_for_unique_seed_mapping": True,
    "public_candidate_observations_generated": False,
    "public_candidate_scene_payloads_retained_or_released": False,
    "public_candidate_fingerprint_from_transient_public_state": True,
    "full_manifest_json_parsed": False,
    "protected_final_scene_identity_or_geometry_access": False,
    "actor_model_loaded_or_inferred": False,
    "actor_bytes_used_for_exact_binding_only": True,
    "actor_outputs_or_labels_released": False,
    "intended_use": "all trace-touched retired v1/v2 identity exclusion only",
}


def producer_sources() -> dict[str, str]:
    """Return the recursive source closure of this isolated producer/reader."""
    return dict(sorted(local_source_hashes((Path(__file__).resolve(),)).items()))


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise ValueError("Exact lowercase SHA-256 required for " + label)
    return value


def _regular(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or path.stat().st_size > MAX_JSON_BYTES):
        raise ValueError(label + " must be a canonical regular file")
    return path


def _parse_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("Duplicate JSON field in " + label)
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("Non-finite JSON value in " + label + ": " + token)
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(label + " cannot be read") from error
    if not isinstance(value, dict):
        raise ValueError(label + " must be one JSON object")
    return value


def _read_bytes_and_sha(path: Path, label: str) -> tuple[bytes, str]:
    """Read and hash one immutable O_NOFOLLOW file descriptor."""
    path = _regular(path, label)
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_JSON_BYTES:
            raise ValueError(label + " must be a bounded regular file")
        chunks: list[bytes] = []
        remaining = MAX_JSON_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > MAX_JSON_BYTES:
            raise ValueError(label + " is oversized")
    finally:
        os.close(descriptor)
    raw_sha256 = _bytes_sha256(raw)
    return raw, raw_sha256


def _read_json(
    path: Path, label: str, *, expected_sha256: str,
) -> dict[str, Any]:
    raw, actual_sha256 = _read_bytes_and_sha(path, label)
    if actual_sha256 != _sha(expected_sha256, label + " bytes"):
        raise ValueError("Exact " + label + " bytes required")
    return _parse_json_bytes(raw, label)


def _identity_rows(rows: Any, *, label: str) -> list[dict[str, Any]]:
    if (not isinstance(rows, list)
            or len(rows) != ACCEPTED_IDENTITIES_PER_SOURCE):
        raise ValueError(label + " must contain exactly 64 scene identities")
    result: list[dict[str, Any]] = []
    seeds: set[int] = set()
    fingerprints: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(label + " scene identity schema differs")
        seed = row.get("seed")
        fingerprint = row.get("fingerprint")
        if (type(seed) is not int or seed < 0
                or type(fingerprint) is not str
                or _HEX.fullmatch(fingerprint) is None
                or seed in seeds or fingerprint in fingerprints):
            raise ValueError(label + " scene identity differs or is duplicated")
        seeds.add(seed)
        fingerprints.add(fingerprint)
        result.append({"seed": seed, "fingerprint": fingerprint})
    result.sort(key=lambda row: (row["seed"], row["fingerprint"]))
    return result


def _candidate_seed_map(manifest: Mapping[str, Any]) -> dict[str, int]:
    """Map authenticated public candidates without retaining scene payloads."""
    batches = manifest.get("candidate_batches")
    authentication = manifest.get("authentication")
    if (not isinstance(batches, list) or len(batches) != 3
            or not isinstance(authentication, Mapping)
            or authentication.get("full_manifest_json_parsed") is not False
            or authentication.get(
                "candidate_scenes_disjoint_from_all_base_splits_authenticated")
                is not True):
        raise ValueError("Authenticated public candidate projection required")
    result: dict[str, int] = {}
    seen_seeds: set[int] = set()
    for batch in batches:
        if not isinstance(batch, list):
            raise ValueError("Public candidate batch schema differs")
        for row in batch:
            if not isinstance(row, Mapping):
                raise ValueError("Public candidate identity schema differs")
            fingerprint = row.get("fingerprint")
            seed = row.get("seed")
            if (type(fingerprint) is not str
                    or _HEX.fullmatch(fingerprint) is None
                    or type(seed) is not int or seed < 0
                    or fingerprint in result or seed in seen_seeds):
                raise ValueError("Public candidate identities are not unique")
            result[fingerprint] = seed
            seen_seeds.add(seed)
    if len(result) != 2160:
        raise ValueError("Exact public candidate population required")
    return result


def _identity_only_reset(
    env: R41DiagnosticConflictWarehouseEnv, *, seed: int,
) -> None:
    """Create the exact candidate reset state without publishing observations.

    The normal online ``reset`` API returns observations (twice through its
    MRO).  Retired-source declassification needs only the deterministic public
    scene identity, so this local boundary reproduces the state-initialization
    portion of ``NativeWarehouseEnv.reset`` and the diagnostic mixin's
    provenance checks.  It deliberately does not call ``reset``, ``_info`` or
    ``observations``.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("Candidate identity seed must be non-negative")
    env._history = None
    env._r41_sampler_draws = 0
    env._r41_creation_history = []
    env._r41_active_task_nodes = {}
    env._r41_sampling_context = None
    env._begin_sampling_batch((), int(env.config.active_task_count))
    try:
        env._rng.seed(seed)
        env._episode_counter += 1
        excluded = {
            env.layout.charger_position,
            *env.layout.robot_start_positions,
            *env.layout.task_endpoint_exclusions,
        }
        tasks = []
        for index in range(1, env.config.active_task_count + 1):
            task = env._sample_delivery_job(
                task_index=index,
                created_frame=0,
                excluded_positions=excluded,
            )
            tasks.append(task)
            excluded.update((task.pickup_position, task.delivery_position))
        agents = [
            AgentState(
                agent_id=agent_id,
                position=env.layout.robot_start_positions[index],
                heading=env._rng.choice(tuple(MOVE_DELTAS)),
            )
            for index, agent_id in enumerate(env.agent_ids)
        ]
        env.state = WarehouseState(
            episode_id=env._episode_counter,
            frame=0,
            agents=agents,
            tasks=tasks,
            next_task_index=len(tasks) + 1,
        )
        created = env._finish_sampling_batch()
    except Exception:
        env._r41_sampling_context = None
        raise
    env._r41_active_task_nodes = {
        task.task_id: task_node_id(task) for task in env.state.tasks
    }
    if {row["task_id"] for row in created} != set(env._r41_active_task_nodes):
        raise RuntimeError("Candidate identity reset provenance differs")
    validate_diagnostic_active_conflict(env.state.tasks, config=env.config)


def _candidate_identity_batch(
    batch_index: int, *, per_family: int, maximum_draws: int = 100_000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate one public seed/fingerprint batch without scene payloads."""
    scenes_api = manifest_binding.scenes_api
    if (type(batch_index) is not int
            or not 0 <= batch_index < scenes_api.MAXIMUM_BATCHES):
        raise ValueError("Candidate identity batch index differs")
    if type(per_family) is not int or per_family < 1:
        raise ValueError("Candidate identity family quota must be positive")
    if type(maximum_draws) is not int or maximum_draws < 1:
        raise ValueError("Candidate identity draw bound must be positive")
    counts = Counter()
    candidates: list[tuple[str, int, str]] = []
    seen_fingerprints: set[str] = set()
    seed = scenes_api.BATCH_SEED_STARTS[batch_index]
    draws = 0
    while any(counts[family] < per_family for family in scenes_api.FAMILY_IDS):
        if draws >= maximum_draws:
            raise RuntimeError(
                "Could not fill every candidate identity family quota")
        current = seed
        seed += 1
        draws += 1
        env = R41DiagnosticConflictWarehouseEnv()
        _identity_only_reset(env, seed=current)
        family_id = conflict_family_id(env.state.tasks)
        if counts[family_id] >= per_family:
            continue
        manifest_binding.scenes_api._candidate_start_state(
            env,
            seed=current,
            batch_index=batch_index,
            family_id=family_id,
        )
        fingerprint = diagnostic_scene_fingerprint(env)
        if fingerprint in seen_fingerprints:
            continue
        seen_fingerprints.add(fingerprint)
        counts[family_id] += 1
        candidates.append((family_id, current, fingerprint))
    family_rank = {
        family: index for index, family in enumerate(scenes_api.FAMILY_IDS)
    }
    candidates.sort(key=lambda row: (family_rank[row[0]], row[1]))
    identities = [
        {"seed": seed_value, "fingerprint": fingerprint}
        for _, seed_value, fingerprint in candidates
    ]
    report = {
        "batch_index": batch_index,
        "seed_start": scenes_api.BATCH_SEED_STARTS[batch_index],
        "draws": draws,
        "candidate_count": len(identities),
        "per_family": dict(sorted(counts.items())),
        "candidate_fingerprints_sha256": digest(
            [row["fingerprint"] for row in identities]
        ),
    }
    return identities, report


def _regenerate_development_candidate_identity_batches(
) -> list[list[dict[str, Any]]]:
    """Regenerate and authenticate only the fixed public identities."""
    scenes_api = manifest_binding.scenes_api
    batches: list[list[dict[str, Any]]] = []
    reports: list[dict[str, Any]] = []
    all_seeds: set[int] = set()
    all_fingerprints: set[str] = set()
    for batch_index in range(scenes_api.MAXIMUM_BATCHES):
        identities, report = _candidate_identity_batch(
            batch_index, per_family=scenes_api.PER_FAMILY_PER_BATCH)
        expected = manifest_binding.EXPECTED_CANDIDATE_BATCHES[batch_index]
        bound_identities = []
        for index, row in enumerate(identities):
            # Candidate order is fixed by family then seed.  Re-derive the
            # public ID solely to authenticate the historical identity digest;
            # IDs are not retained or returned by this boundary.
            family_id = scenes_api.FAMILY_IDS[
                index // scenes_api.PER_FAMILY_PER_BATCH
            ]
            offset = index % scenes_api.PER_FAMILY_PER_BATCH
            bound_identities.append({
                "id": (
                    f"diagnostic_b{batch_index}_{family_id}_{offset:03d}"
                ),
                "seed": row["seed"],
                "fingerprint": row["fingerprint"],
            })
        seeds = {row["seed"] for row in identities}
        fingerprints = {row["fingerprint"] for row in identities}
        if (
            len(identities) != expected["count"]
            or digest(bound_identities) != expected["identity_sha256"]
            or digest(report) != expected["report_sha256"]
            or len(seeds) != len(identities)
            or len(fingerprints) != len(identities)
            or all_seeds & seeds
            or all_fingerprints & fingerprints
        ):
            raise ValueError(
                "Regenerated candidate identity batch differs from its binding")
        all_seeds.update(seeds)
        all_fingerprints.update(fingerprints)
        batches.append(identities)
        reports.append(report)
    if digest(reports) != manifest_binding.EXPECTED_CANDIDATE_REPORTS_SHA256:
        raise ValueError("Regenerated candidate identity reports differ")
    return batches


def _declassify(
    values: Mapping[str, tuple[str, Mapping[str, Any]]],
    *, candidate_seeds: Mapping[str, int],
) -> dict[str, Any]:
    """Project accepted and trace-touched scenes onto exposure identities."""
    if set(values) != set(SOURCE_FILE_SHA256):
        raise ValueError("Exactly one fixed retired v1 and v2 source is required")
    sources: list[dict[str, Any]] = []
    accepted_seeds: set[int] = set()
    accepted_fingerprints: set[str] = set()
    global_exposed_seeds: set[int] = set()
    global_exposed_fingerprints: set[str] = set()
    for version in sorted(SOURCE_FILE_SHA256):
        source_sha256, value = values[version]
        if (value.get("version") != version
                or source_sha256 != SOURCE_FILE_SHA256[version]):
            raise ValueError("Retired source version or exact bytes differ")
        accepted = _identity_rows(
            value.get("scenes"), label="retired " + version)
        source_accepted_seeds = {row["seed"] for row in accepted}
        source_accepted_fingerprints = {
            row["fingerprint"] for row in accepted}
        if (accepted_seeds & source_accepted_seeds
                or accepted_fingerprints & source_accepted_fingerprints):
            raise ValueError("Accepted retired identities overlap across versions")
        accepted_seeds.update(source_accepted_seeds)
        accepted_fingerprints.update(source_accepted_fingerprints)

        trace = value.get("selection_trace")
        if not isinstance(trace, list) or not trace:
            raise ValueError("Retired selection exposure trace is missing")
        touched: set[str] = set()
        for row in trace:
            if not isinstance(row, Mapping):
                raise ValueError("Retired selection exposure trace schema differs")
            fingerprint = row.get("fingerprint")
            if (type(fingerprint) is not str
                    or _HEX.fullmatch(fingerprint) is None):
                raise ValueError("Retired selection exposure fingerprint differs")
            touched.add(fingerprint)
        exposed_fingerprints = touched | source_accepted_fingerprints
        expected_count = EXPECTED_EXPOSED_IDENTITIES_PER_SOURCE[version]
        if len(exposed_fingerprints) != expected_count:
            raise ValueError("Retired selection exposure count differs")
        if any(fingerprint not in candidate_seeds
               for fingerprint in exposed_fingerprints):
            raise ValueError(
                "Retired exposure fingerprint is absent from the public candidate pool")
        for row in accepted:
            if candidate_seeds[row["fingerprint"]] != row["seed"]:
                raise ValueError(
                    "Accepted retired seed differs from the public candidate pool")
        source_exposed_seeds = {
            candidate_seeds[fingerprint] for fingerprint in exposed_fingerprints}
        if len(source_exposed_seeds) != len(exposed_fingerprints):
            raise ValueError("One public candidate seed maps to two exposures")
        global_exposed_seeds.update(source_exposed_seeds)
        global_exposed_fingerprints.update(exposed_fingerprints)
        sources.append({
            "version": version,
            "source_file_sha256": source_sha256,
            "accepted_count": len(accepted),
            "exposed_count": len(exposed_fingerprints),
        })
    if (len(global_exposed_fingerprints) != EXPECTED_GLOBAL_EXPOSED_IDENTITIES
            or len(global_exposed_seeds) != EXPECTED_GLOBAL_EXPOSED_IDENTITIES):
        raise ValueError("Global retired exposure identity count differs")
    exposed_identities = sorted(
        ({"seed": candidate_seeds[fingerprint], "fingerprint": fingerprint}
         for fingerprint in global_exposed_fingerprints),
        key=lambda row: (row["seed"], row["fingerprint"]),
    )
    projection: dict[str, Any] = {
        "version": VERSION,
        "sources": sources,
        "exposed_identities": exposed_identities,
    }
    projection["content_sha256"] = digest(projection)
    return projection


def _validate_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    if set(value) != _PROJECTION_FIELDS or value.get("version") != VERSION:
        raise ValueError("Retired identity projection top-level schema differs")
    claimed = _sha(value.get("content_sha256"), "projection content")
    if claimed != digest({key: child for key, child in value.items()
                          if key != "content_sha256"}):
        raise ValueError("Retired identity projection content hash differs")
    rows = value.get("sources")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("Retired identity projection requires two sources")
    versions: set[str] = set()
    all_seeds: set[int] = set()
    all_fingerprints: set[str] = set()
    for source in rows:
        if not isinstance(source, Mapping) or set(source) != _SOURCE_FIELDS:
            raise ValueError("Retired identity projection source schema differs")
        version = source.get("version")
        if (version not in SOURCE_FILE_SHA256 or version in versions
                or source.get("source_file_sha256")
                    != SOURCE_FILE_SHA256.get(str(version))):
            raise ValueError("Retired identity projection source binding differs")
        versions.add(str(version))
        if source.get("accepted_count") != ACCEPTED_IDENTITIES_PER_SOURCE:
            raise ValueError("Retired accepted-count audit differs")
        if (source.get("exposed_count")
                != EXPECTED_EXPOSED_IDENTITIES_PER_SOURCE[str(version)]):
            raise ValueError("Retired identity projection count differs")
    if versions != set(SOURCE_FILE_SHA256):
        raise ValueError("Retired identity projection source set differs")
    identities = value.get("exposed_identities")
    if (not isinstance(identities, list)
            or len(identities) != EXPECTED_GLOBAL_EXPOSED_IDENTITIES):
        raise ValueError("Retired identity projection global count differs")
    previous: tuple[int, str] | None = None
    for identity in identities:
        if not isinstance(identity, Mapping) or set(identity) != _IDENTITY_FIELDS:
            raise ValueError("Retired identity projection leaked extra fields")
        seed = identity.get("seed")
        fingerprint = identity.get("fingerprint")
        if (type(seed) is not int or seed < 0
                or type(fingerprint) is not str
                or _HEX.fullmatch(fingerprint) is None
                or seed in all_seeds or fingerprint in all_fingerprints
                or (previous is not None and (seed, fingerprint) <= previous)):
            raise ValueError("Retired identity projection identity differs")
        all_seeds.add(seed)
        all_fingerprints.add(fingerprint)
        previous = (seed, fingerprint)
    if (len(all_seeds) != EXPECTED_GLOBAL_EXPOSED_IDENTITIES
            or len(all_fingerprints) != EXPECTED_GLOBAL_EXPOSED_IDENTITIES):
        raise ValueError("Retired identity projection global exposure differs")
    return deepcopy(dict(value))


def _raw_json(value: Mapping[str, Any]) -> bytes:
    return (canonical(value) + "\n").encode("utf-8")


def _bytes_sha256(value: bytes) -> str:
    return sha256(value).hexdigest()


def _write_exclusive(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink() or path.parent.is_symlink():
        raise ValueError("Retired identity projection output path is unsafe")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_output(
    output: Path, projection_raw: bytes, report_raw: bytes, *,
    before_publish: Callable[[], None],
) -> None:
    parent = output.parent
    if (not parent.is_dir() or parent.is_symlink() or parent.resolve() != parent
            or output.exists() or output.is_symlink()):
        raise ValueError("Retired identity projection destination is unsafe")
    lock = parent / ("." + output.name + ".lock")
    lock_fd = None
    temporary: Path | None = None
    try:
        lock_fd = os.open(
            lock,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        temporary = Path(tempfile.mkdtemp(
            prefix="." + output.name + ".tmp-", dir=parent)).absolute()
        os.chmod(temporary, 0o700)
        _write_exclusive(
            temporary / "retired_identity_projection.json", projection_raw)
        _write_exclusive(temporary / "report.json", report_raw)
        directory_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        before_publish()
        if output.exists() or output.is_symlink():
            raise ValueError("Retired identity projection destination appeared")
        os.rename(temporary, output)
        temporary = None
        parent_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        if lock_fd is not None:
            os.close(lock_fd)
            lock.unlink(missing_ok=True)


def _report(
    projection: Mapping[str, Any], projection_file_sha256: str,
    source_files: Mapping[str, str], producer: Mapping[str, str],
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "version": REPORT_VERSION,
        "status": STATUS,
        "projection_file_sha256": projection_file_sha256,
        "projection_content_sha256": projection["content_sha256"],
        "source_file_sha256": dict(sorted(source_files.items())),
        "identity_counts": {
            source["version"]: {
                "accepted": source["accepted_count"],
                "exposed": source["exposed_count"],
            }
            for source in projection["sources"]
        },
        "global_identity_count": EXPECTED_GLOBAL_EXPOSED_IDENTITIES,
        "identity_uniqueness": {
            "seed": EXPECTED_GLOBAL_EXPOSED_IDENTITIES,
            "fingerprint": EXPECTED_GLOBAL_EXPOSED_IDENTITIES,
        },
        "declassification": deepcopy(_DECLASSIFICATION),
        "actor_file_sha256": EXPECTED_ACTOR_SHA256,
        "manifest_file_sha256": EXPECTED_MANIFEST_SHA256,
        "manifest_validation_file_sha256": (
            EXPECTED_MANIFEST_VALIDATION_SHA256),
        "development_projection_sha256": (
            manifest_binding.EXPECTED_DEVELOPMENT_PROJECTION_SHA256),
        "supersedes_projection_sha256": SUPERSEDES_PROJECTION_SHA256,
        "producer_sources": dict(sorted(producer.items())),
        "producer_sources_sha256": digest(producer),
    }
    value["content_sha256"] = digest(value)
    return value


def _validate_report(
    value: Mapping[str, Any], *, projection: Mapping[str, Any],
    projection_file_sha256: str, report_file_sha256: str,
    expected_report_sha256: str, sources: Mapping[str, str],
) -> dict[str, Any]:
    if report_file_sha256 != expected_report_sha256:
        raise ValueError("Exact retired identity projection audit bytes required")
    if (set(value) != _REPORT_FIELDS
            or value.get("version") != REPORT_VERSION
            or value.get("status") != STATUS
            or value.get("projection_file_sha256") != projection_file_sha256
            or value.get("projection_content_sha256")
                != projection.get("content_sha256")
            or value.get("source_file_sha256")
                != dict(sorted(SOURCE_FILE_SHA256.items()))
            or value.get("identity_counts")
                != {
                    version: {
                        "accepted": ACCEPTED_IDENTITIES_PER_SOURCE,
                        "exposed": EXPECTED_EXPOSED_IDENTITIES_PER_SOURCE[version],
                    }
                    for version in SOURCE_FILE_SHA256
                }
            or value.get("global_identity_count")
                != EXPECTED_GLOBAL_EXPOSED_IDENTITIES
            or value.get("identity_uniqueness")
                != {
                    "seed": EXPECTED_GLOBAL_EXPOSED_IDENTITIES,
                    "fingerprint": EXPECTED_GLOBAL_EXPOSED_IDENTITIES,
                }
            or value.get("declassification") != _DECLASSIFICATION
            or value.get("actor_file_sha256") != EXPECTED_ACTOR_SHA256
            or value.get("manifest_file_sha256") != EXPECTED_MANIFEST_SHA256
            or value.get("manifest_validation_file_sha256")
                != EXPECTED_MANIFEST_VALIDATION_SHA256
            or value.get("development_projection_sha256")
                != manifest_binding.EXPECTED_DEVELOPMENT_PROJECTION_SHA256
            or value.get("supersedes_projection_sha256")
                != SUPERSEDES_PROJECTION_SHA256
            or value.get("producer_sources") != dict(sorted(sources.items()))
            or value.get("producer_sources_sha256") != digest(sources)):
        raise ValueError("Retired identity projection audit binding differs")
    claimed = _sha(value.get("content_sha256"), "projection audit content")
    if claimed != digest({key: child for key, child in value.items()
                          if key != "content_sha256"}):
        raise ValueError("Retired identity projection audit content hash differs")
    return deepcopy(dict(value))


def build(
    *, actor_path: str | Path, manifest_path: str | Path,
    retired_holdout_paths: Sequence[str | Path], output: str | Path,
) -> dict[str, Any]:
    """Build one deterministic identity-only projection from fixed v1/v2."""
    sources = producer_sources()
    if len(retired_holdout_paths) != 2:
        raise ValueError("Exactly two fixed retired holdouts are required")
    actor_file = _regular(actor_path, "frozen diagnostic Actor")
    manifest_file = _regular(manifest_path, "frozen diagnostic manifest")
    validation_file = _regular(
        manifest_file.parent / "validation.json",
        "frozen diagnostic manifest validation",
    )
    paths = [_regular(value, "retired holdout")
             for value in retired_holdout_paths]
    if len(set(paths)) != 2:
        raise ValueError("Retired holdout inputs must be distinct")
    output_path = Path(output).expanduser().absolute()

    # The private retired JSON and all values derived from it live only in this
    # nested worker.  The public wrapper destroys the worker and its exception
    # traceback before raising one fixed error, so callers cannot recover raw
    # scenes, trace rows, or exposure identities through frame locals.
    def _sensitive_worker() -> dict[str, Any]:
        # Open each private retired source exactly once, hash the bytes from
        # that same O_NOFOLLOW descriptor, and authenticate before parsing.
        source_bytes: dict[Path, bytes] = {}
        source_hashes: dict[Path, str] = {}
        for path in paths:
            raw, actual_sha256 = _read_bytes_and_sha(path, "retired holdout")
            if actual_sha256 not in SOURCE_FILE_SHA256.values():
                raise ValueError("Exact retired v1/v2 source bytes are required")
            source_bytes[path] = raw
            source_hashes[path] = actual_sha256
        if set(source_hashes.values()) != set(SOURCE_FILE_SHA256.values()):
            raise ValueError("Exact retired v1/v2 source bytes are required")
        with ImmutableInputSnapshot(
            {
                "actor": actor_file,
                "manifest": manifest_file,
                "manifest_validation": validation_file,
            },
            expected_sha256={
                "actor": EXPECTED_ACTOR_SHA256,
                "manifest": EXPECTED_MANIFEST_SHA256,
                "manifest_validation": EXPECTED_MANIFEST_VALIDATION_SHA256,
            },
            relative_names={
                "manifest": "manifest/manifest.json",
                "manifest_validation": "manifest/validation.json",
            },
            prefix="warehouse-r41-retired-projection-inputs-",
        ) as frozen:
            if producer_sources() != sources:
                raise RuntimeError(
                    "Retired projection sources changed before authentication")
            manifest_identity = manifest_binding.read_saved_manifest(
                frozen.entries["manifest"].frozen,
                expected_sha256=EXPECTED_MANIFEST_SHA256,
                replay_scope="none",
            )
            authentication = manifest_identity.get("authentication", {})
            if (authentication.get("full_manifest_json_parsed") is not False
                    or authentication.get("replay_scope") != "none"
                    or authentication.get(
                        "candidate_scenes_disjoint_from_all_base_splits_authenticated")
                        is not True):
                raise ValueError("Exact identity-only manifest authentication required")
            candidate_identity_batches = (
                _regenerate_development_candidate_identity_batches())
            candidate_seeds = _candidate_seed_map({
                "authentication": authentication,
                "candidate_batches": candidate_identity_batches,
            })

            values: dict[str, tuple[str, Mapping[str, Any]]] = {}
            for path in paths:
                value = _parse_json_bytes(source_bytes[path], "retired holdout")
                version = value.get("version")
                source_sha256 = source_hashes[path]
                if (version not in SOURCE_FILE_SHA256 or version in values
                        or SOURCE_FILE_SHA256.get(str(version)) != source_sha256):
                    raise ValueError("Exactly one fixed retired v1 and v2 is required")
                values[str(version)] = (source_sha256, value)
            projection = _declassify(values, candidate_seeds=candidate_seeds)
            projection_raw = _raw_json(projection)
            projection_file_sha256 = _bytes_sha256(projection_raw)
            report = _report(
                projection,
                projection_file_sha256,
                {version: source_sha256
                 for version, (source_sha256, _) in values.items()},
                sources,
            )
            report_raw = _raw_json(report)

            def verify_frozen_inputs() -> None:
                frozen.verify()
                if (producer_sources() != sources
                        or any(file_hash(path) != expected
                               for path, expected in source_hashes.items())):
                    raise RuntimeError(
                        "Retired projection sources or inputs changed during build")

            verify_frozen_inputs()
            _atomic_output(
                output_path, projection_raw, report_raw,
                before_publish=verify_frozen_inputs,
            )
            try:
                verify_frozen_inputs()
                if (file_hash(output_path / "retired_identity_projection.json")
                        != projection_file_sha256
                        or file_hash(output_path / "report.json")
                            != _bytes_sha256(report_raw)):
                    raise RuntimeError(
                        "Published retired identity projection bytes differ")
            except BaseException:
                shutil.rmtree(output_path, ignore_errors=True)
                raise
        return deepcopy(report)

    result: dict[str, Any] | None = None
    failed = False
    try:
        result = _sensitive_worker()
    except BaseException as private_failure:
        try:
            shutil.rmtree(output_path, ignore_errors=True)
        except BaseException:
            pass
        try:
            private_failure.__traceback__ = None
            private_failure.__cause__ = None
            private_failure.__context__ = None
        except BaseException:
            pass
        failed = True
    if failed:
        _sensitive_worker = None
        result = None
        sources = {}
        paths = []
        actor_file = manifest_file = validation_file = output_path = None
        actor_path = manifest_path = output = None
        retired_holdout_paths = ()
        raise RuntimeError("Retired identity projection private build failed") from None
    _sensitive_worker = None
    if result is None:
        raise AssertionError("Retired identity projection worker returned no report")
    return result


def read_saved_projection(
    path: str | Path, *, expected_projection_sha256: str,
    expected_report_sha256: str,
) -> dict[str, Any]:
    """Strictly authenticate a saved projection and its canonical audit."""
    sources = producer_sources()
    expected_projection_sha256 = _sha(
        expected_projection_sha256, "saved projection file")
    expected_report_sha256 = _sha(
        expected_report_sha256, "saved projection audit file")
    projection_path = _regular(path, "retired identity projection")
    if projection_path.name != "retired_identity_projection.json":
        raise ValueError("Canonical retired identity projection filename required")
    report_path = _regular(
        projection_path.parent / "report.json",
        "retired identity projection audit",
    )
    if producer_sources() != sources:
        raise RuntimeError(
            "Retired projection sources changed before artifact read")
    projection = _validate_projection(
        _read_json(
            projection_path,
            "retired identity projection",
            expected_sha256=expected_projection_sha256,
        ))
    projection_sha256 = expected_projection_sha256
    report = _read_json(
        report_path,
        "retired identity projection audit",
        expected_sha256=expected_report_sha256,
    )
    report_sha256 = expected_report_sha256
    _validate_report(
        report,
        projection=projection,
        projection_file_sha256=projection_sha256,
        report_file_sha256=report_sha256,
        expected_report_sha256=expected_report_sha256,
        sources=sources,
    )
    if (producer_sources() != sources
            or file_hash(projection_path) != projection_sha256
            or file_hash(report_path) != report_sha256):
        raise RuntimeError(
            "Retired projection source or artifact changed during read")
    return projection


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--retired-holdout", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(canonical(build(
        actor_path=args.actor,
        manifest_path=args.manifest,
        retired_holdout_paths=args.retired_holdout,
        output=args.output,
    )))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VERSION", "REPORT_VERSION", "STATUS", "ACCEPTED_IDENTITIES_PER_SOURCE",
    "EXPECTED_EXPOSED_IDENTITIES_PER_SOURCE",
    "EXPECTED_GLOBAL_EXPOSED_IDENTITIES",
    "SOURCE_FILE_SHA256", "producer_sources", "build",
    "read_saved_projection", "main",
]
