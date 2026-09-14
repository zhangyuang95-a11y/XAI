"""Build the public candidate universe for the v14 protected final.

The v13 program and its passed outer evaluation remain frozen.  This module
only creates new public scene identities after authenticating the permanent
v13 timeout closeout.  It never reads either final salt, an action label,
Actor probabilities, or program predictions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from backend.training import warehouse_r41_diagnostic_conflict_scenarios as scenes_api
from backend.training import warehouse_r41_diagnostic_final_timeout_closeout_public_v14 as closeout_api
from backend.training.warehouse_diagnostic_source_closure import local_source_hashes
from backend.training.warehouse_native_common import canonical, digest, file_hash


VERSION = "warehouse-r41-diagnostic-final-candidate-universe.v14"
STATUS = "frozen_public_candidate_universe"
SEED_STARTS = (58_100_000, 58_200_000, 58_300_000)
BATCH_MARKERS = (14, 15, 16)
PER_FAMILY_PER_BATCH = 120
FAMILY_IDS = tuple(scenes_api.FAMILY_IDS)
BATCH_COUNT = len(SEED_STARTS)
SCENE_COUNT = BATCH_COUNT * PER_FAMILY_PER_BATCH * len(FAMILY_IDS)
MAXIMUM_DRAWS_PER_BATCH = 100_000
MAX_JSON_BYTES = 512 * 1024 * 1024
ROOT = Path(__file__).resolve().parents[2]
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def producer_sources() -> dict[str, str]:
    return dict(sorted(local_source_hashes((Path(__file__).resolve(),)).items()))


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "seed_starts": list(SEED_STARTS),
        "batch_markers": list(BATCH_MARKERS),
        "per_family_per_batch": PER_FAMILY_PER_BATCH,
        "batch_count": BATCH_COUNT,
        "scene_count": SCENE_COUNT,
        "family_ids": list(FAMILY_IDS),
        "scene_generator": "diagnostic-v3 public candidate start",
        "old_ranked_identity_population_excluded": True,
        "historical_exposed_identity_population_excluded": True,
        "v13_outer_identity_population_excluded": True,
        "timeout_closeout_authenticated_before_generation": True,
        "program_access": False,
        "program_predictions_access": False,
        "action_labels_access": False,
        "actor_probabilities_access": False,
        "private_salt_access": False,
        "protected_final_material_access": False,
    }


def _content_valid(value: Mapping[str, Any]) -> bool:
    content = value.get("content_sha256")
    return (type(content) is str and _HEX.fullmatch(content) is not None
            and content == digest({
                name: child for name, child in value.items()
                if name != "content_sha256"}))


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise ValueError(label + " must be a SHA-256 digest")
    return value


def _candidate_identity(row: Mapping[str, Any], label: str) -> dict[str, Any]:
    required = {"batch_index", "family_offset", "family_id", "seed", "fingerprint"}
    if (set(row) != required or type(row.get("batch_index")) is not int
            or type(row.get("family_offset")) is not int
            or row.get("family_id") not in FAMILY_IDS
            or type(row.get("seed")) is not int or row["seed"] < 0
            or type(row.get("fingerprint")) is not str
            or _HEX.fullmatch(row["fingerprint"]) is None):
        raise ValueError(label + " identity differs")
    return {name: row[name] for name in (
        "batch_index", "family_offset", "family_id", "seed", "fingerprint")}


def _old_identity(row: Mapping[str, Any], label: str) -> dict[str, Any]:
    required = {
        "batch_index", "family_offset", "family_id", "seed", "fingerprint"}
    if (set(row) != required or type(row.get("batch_index")) is not int
            or row.get("batch_index") < 0
            or type(row.get("family_offset")) is not int
            or row.get("family_offset") < 0
            or row.get("family_id") not in FAMILY_IDS
            or type(row.get("seed")) is not int or row["seed"] < 0
            or type(row.get("fingerprint")) is not str
            or _HEX.fullmatch(row["fingerprint"]) is None):
        raise ValueError(label + " identity differs")
    return {name: row[name] for name in (
        "batch_index", "family_offset", "family_id", "seed", "fingerprint")}


def _sorted_ints(values: Any, label: str) -> list[int]:
    if (not isinstance(values, list) or any(type(value) is not int or value < 0
                                            for value in values)
            or values != sorted(set(values))):
        raise ValueError(label + " differs")
    return list(values)


def _sorted_hashes(values: Any, label: str) -> list[str]:
    if (not isinstance(values, list)
            or any(type(value) is not str or _HEX.fullmatch(value) is None
                   for value in values)
            or values != sorted(set(values))):
        raise ValueError(label + " differs")
    return list(values)


def _closeout_exclusions(
    closeout: Mapping[str, Any], *, closeout_path: Path,
) -> dict[str, Any]:
    """Return the strict public identity/hash boundary frozen by the closeout."""
    bindings = closeout.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("V13 timeout closeout lacks its public burn boundary")
    _burned_path, burned = _strict_json(
        closeout_path.parent / closeout_api.IDENTITY_NAME,
        "v13 burned candidate universe",
        expected_sha256=_sha(
            bindings.get("burned_candidate_universe_file_sha256"),
            "v13 burned candidate universe"))
    _projection_path, projection = _strict_json(
        closeout_path.parent / closeout_api.PROJECTION_NAME,
        "v13 burned observation projection",
        expected_sha256=_sha(
            bindings.get("burned_observation_hashes_file_sha256"),
            "v13 burned observation projection"))
    if (burned.get("content_sha256")
            != bindings.get("burned_candidate_universe_content_sha256")
            or projection.get("content_sha256")
                != bindings.get("burned_observation_hashes_content_sha256")):
        raise ValueError("V13 timeout closeout companion binding differs")
    ranked = burned.get("ranking_input_identities")
    if not isinstance(ranked, list):
        raise ValueError("V13 ranked identity population differs")
    ranked_identities = [_old_identity(row, "v13 ranked") for row in ranked]
    if (len(ranked_identities) != 2160
            or len({row["seed"] for row in ranked_identities}) != 2160
            or len({row["fingerprint"] for row in ranked_identities}) != 2160
            or burned.get("ranking_input_identities_sha256")
                != digest(ranked_identities)):
        raise ValueError("V13 ranked identity population binding differs")
    historical_seeds = _sorted_ints(
        burned.get("historically_excluded_seeds"), "historical excluded seeds")
    historical_fingerprints = _sorted_hashes(
        burned.get("historically_excluded_fingerprints"),
        "historical excluded fingerprints")
    if (burned.get("historically_excluded_seeds_sha256")
            != digest(historical_seeds)
            or burned.get("historically_excluded_fingerprints_sha256")
                != digest(historical_fingerprints)):
        raise ValueError("Historical exclusion binding differs")
    burned_hashes = _sorted_hashes(
        projection.get("unique_observation_hashes"),
        "burned v13 observation hashes")
    if (projection.get("unique_observation_hashes_sha256")
            != digest(burned_hashes)):
        raise ValueError("Burned v13 observation projection differs")
    return {
        "ranked_identities": ranked_identities,
        "ranked_identities_sha256": digest(ranked_identities),
        "historical_excluded_seeds": historical_seeds,
        "historical_excluded_seeds_sha256": digest(historical_seeds),
        "historical_excluded_fingerprints": historical_fingerprints,
        "historical_excluded_fingerprints_sha256": digest(
            historical_fingerprints),
        "burned_observation_hashes_sha256": digest(burned_hashes),
        "burned_unique_observation_count": len(burned_hashes),
    }


def _generate_candidate_batches(
    *, excluded_seeds: set[int], excluded_fingerprints: set[str],
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    seen_seeds = set(excluded_seeds)
    seen_fingerprints = set(excluded_fingerprints)
    batches: list[list[dict[str, Any]]] = []
    reports: list[dict[str, Any]] = []
    for logical_batch, (seed_start, marker) in enumerate(
            zip(SEED_STARTS, BATCH_MARKERS)):
        counts: Counter[str] = Counter()
        rows: list[dict[str, Any]] = []
        seed = seed_start
        draws = 0
        while any(counts[family] < PER_FAMILY_PER_BATCH
                  for family in FAMILY_IDS):
            if draws >= MAXIMUM_DRAWS_PER_BATCH:
                raise RuntimeError("V14 candidate family quota unavailable")
            current = seed
            seed += 1
            draws += 1
            scene = scenes_api.make_scene(
                f"diagnostic_v14_probe_{logical_batch}_{draws:06d}",
                "v14_final_candidates", current,
                batch_index=marker, candidate_start=True)
            family = scene.get("family_id")
            fingerprint = scene.get("fingerprint")
            if family not in FAMILY_IDS:
                raise RuntimeError("V14 candidate family differs")
            if (counts[family] >= PER_FAMILY_PER_BATCH
                    or current in seen_seeds
                    or fingerprint in seen_fingerprints):
                continue
            offset = counts[family]
            counts[family] += 1
            seen_seeds.add(current)
            seen_fingerprints.add(str(fingerprint))
            scene = deepcopy(dict(scene))
            scene["id"] = (
                f"diagnostic_v14_b{logical_batch}_{family}_{offset:03d}")
            rows.append(scene)
        rows.sort(key=lambda row: (row["family_id"], row["seed"]))
        expected = {family: PER_FAMILY_PER_BATCH for family in FAMILY_IDS}
        if len(rows) != PER_FAMILY_PER_BATCH * len(FAMILY_IDS) \
                or dict(counts) != expected:
            raise RuntimeError("V14 candidate batch accounting differs")
        batches.append(rows)
        reports.append({
            "batch_index": logical_batch,
            "candidate_batch_marker": marker,
            "seed_start": seed_start,
            "last_seed_examined": seed - 1,
            "draws": draws,
            "candidate_count": len(rows),
            "per_family": dict(sorted(counts.items())),
            "candidate_fingerprints_sha256": digest([
                row["fingerprint"] for row in rows]),
        })
    return batches, reports


def _candidate_identities(
    batches: Sequence[Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    counters: Counter[tuple[int, str]] = Counter()
    for logical_batch, rows in enumerate(batches):
        marker = BATCH_MARKERS[logical_batch]
        for raw in rows:
            family = str(raw["family_id"])
            offset = counters[(logical_batch, family)]
            counters[(logical_batch, family)] += 1
            result.append(_candidate_identity({
                "batch_index": marker,
                "family_offset": offset,
                "family_id": family,
                "seed": raw["seed"],
                "fingerprint": raw["fingerprint"],
            }, "v14 candidate"))
    return result


def create_universe(
    *, timeout_closeout_path: str | Path,
    expected_timeout_closeout_sha256: str,
    permanent_timeout_closeout_registry: str | Path,
) -> dict[str, Any]:
    closeout = closeout_api.read_saved_closeout_public(
        timeout_closeout_path,
        expected_closeout_sha256=_sha(
            expected_timeout_closeout_sha256, "v13 timeout closeout"),
        permanent_closeout_registry=permanent_timeout_closeout_registry)
    closeout_path = Path(timeout_closeout_path).expanduser().absolute()
    exclusions = _closeout_exclusions(closeout, closeout_path=closeout_path)
    ranked = exclusions["ranked_identities"]
    excluded_seeds = (
        {row["seed"] for row in ranked}
        | set(exclusions["historical_excluded_seeds"]))
    excluded_fingerprints = (
        {row["fingerprint"] for row in ranked}
        | set(exclusions["historical_excluded_fingerprints"]))
    batches, reports = _generate_candidate_batches(
        excluded_seeds=excluded_seeds,
        excluded_fingerprints=excluded_fingerprints)
    scenes = [deepcopy(dict(scene)) for batch in batches for scene in batch]
    identities = _candidate_identities(batches)
    if (len(scenes) != SCENE_COUNT or len(identities) != SCENE_COUNT
            or len({row["seed"] for row in identities}) != SCENE_COUNT
            or len({row["fingerprint"] for row in identities}) != SCENE_COUNT):
        raise RuntimeError("V14 candidate universe accounting differs")
    new_seeds = {row["seed"] for row in identities}
    new_fingerprints = {row["fingerprint"] for row in identities}
    if new_seeds & excluded_seeds or new_fingerprints & excluded_fingerprints:
        raise RuntimeError("V14 candidate universe overlaps a retired identity")
    family_counts = dict(sorted(Counter(
        row["family_id"] for row in identities).items()))
    expected_counts = {
        family: BATCH_COUNT * PER_FAMILY_PER_BATCH for family in FAMILY_IDS}
    if family_counts != expected_counts:
        raise RuntimeError("V14 candidate universe family balance differs")
    sources = producer_sources()
    bindings = closeout.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError("V13 timeout closeout bindings differ")
    value: dict[str, Any] = {
        "version": VERSION,
        "status": STATUS,
        "contract": contract(),
        "bindings": {
            "timeout_closeout_file_sha256": file_hash(
                Path(timeout_closeout_path).expanduser().absolute()),
            "timeout_closeout_content_sha256": closeout["content_sha256"],
            "actor_sha256": _sha(bindings.get("actor_sha256"), "Actor"),
            "protocol_sha256": _sha(
                bindings.get("protocol_sha256"), "protocol"),
            "runtime_manifest_sha256": _sha(
                bindings.get("runtime_manifest_sha256"), "runtime manifest"),
            "burned_ranked_identities_sha256": exclusions[
                "ranked_identities_sha256"],
            "burned_observation_hashes_sha256": exclusions[
                "burned_observation_hashes_sha256"],
            "historical_excluded_seeds_sha256": exclusions[
                "historical_excluded_seeds_sha256"],
            "historical_excluded_fingerprints_sha256": exclusions[
                "historical_excluded_fingerprints_sha256"],
        },
        "candidate_scenes": scenes,
        "candidate_identities": identities,
        "scene_count": len(scenes),
        "family_counts": family_counts,
        "ordered_identity_sha256": digest(identities),
        "generation_reports": reports,
        "generation_reports_sha256": digest(reports),
        "disjointness": {
            "old_ranked_seed_overlap": len(new_seeds & {
                row["seed"] for row in ranked}),
            "old_ranked_fingerprint_overlap": len(new_fingerprints & {
                row["fingerprint"] for row in ranked}),
            "historical_seed_overlap": len(new_seeds & set(
                exclusions["historical_excluded_seeds"])),
            "historical_fingerprint_overlap": len(new_fingerprints & set(
                exclusions["historical_excluded_fingerprints"])),
            "unique_seeds": len(new_seeds),
            "unique_fingerprints": len(new_fingerprints),
            "passed": True,
        },
        "information_boundary": {
            "timeout_closeout_authenticated": True,
            "program_access": False,
            "program_predictions_access": False,
            "action_labels_access": False,
            "actor_probabilities_access": False,
            "private_salt_access": False,
            "protected_final_material_access": False,
            "participant_data_access": False,
        },
        "producer_sources": sources,
        "producer_sources_sha256": digest(sources),
        "formal_ready": False,
    }
    value["content_sha256"] = digest(value)
    if producer_sources() != sources:
        raise RuntimeError("V14 candidate universe source changed")
    return value


def _directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    if not path.is_dir() or path.is_symlink() or path.resolve() != path:
        raise ValueError(label + " must be a canonical directory")
    return path


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write((canonical(value) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def build(*, output: str | Path, **kwargs: Any) -> dict[str, Any]:
    value = create_universe(**kwargs)
    destination = Path(output).expanduser().absolute()
    parent = _directory(destination.parent, "v14 universe output parent")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("V14 candidate universe output already exists")
    temporary = Path(tempfile.mkdtemp(
        prefix="." + destination.name + ".tmp-", dir=parent)).absolute()
    temporary_file = temporary / destination.name
    try:
        _write_exclusive(temporary_file, value)
        # Hard-link publication gives the destination O_EXCL semantics while
        # making the complete, fsynced bytes visible in one namespace step.
        os.link(temporary_file, destination, follow_symlinks=False)
        directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return deepcopy(value)


def _strict_json(path: str | Path, label: str, *, expected_sha256: str,
                 maximum: int = MAX_JSON_BYTES) -> tuple[Path, dict[str, Any]]:
    candidate = Path(path).expanduser().absolute()
    if (not candidate.is_file() or candidate.is_symlink()
            or candidate.resolve() != candidate
            or candidate.stat(follow_symlinks=False).st_size > maximum
            or file_hash(candidate) != _sha(expected_sha256, label)):
        raise ValueError("Exact " + label + " bytes required")
    raw = candidate.read_bytes()
    parsed = json.loads(raw)
    if (not isinstance(parsed, Mapping)
            or raw != (canonical(parsed) + "\n").encode("utf-8")):
        raise ValueError(label + " must be canonical JSON")
    return candidate, dict(parsed)


def read_saved_universe(
    path: str | Path, *, expected_universe_sha256: str,
    timeout_closeout_path: str | Path,
    expected_timeout_closeout_sha256: str,
    permanent_timeout_closeout_registry: str | Path,
) -> dict[str, Any]:
    universe_path, saved = _strict_json(
        path, "v14 candidate universe",
        expected_sha256=expected_universe_sha256)
    if (saved.get("version") != VERSION or saved.get("status") != STATUS
            or saved.get("contract") != contract()
            or not _content_valid(saved)
            or saved.get("formal_ready") is not False):
        raise ValueError("Saved v14 candidate universe semantics differ")
    recreated = create_universe(
        timeout_closeout_path=timeout_closeout_path,
        expected_timeout_closeout_sha256=expected_timeout_closeout_sha256,
        permanent_timeout_closeout_registry=permanent_timeout_closeout_registry)
    if saved != recreated or file_hash(universe_path) != expected_universe_sha256:
        raise ValueError("Saved v14 candidate universe differs from replay")
    return deepcopy(saved)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-closeout", required=True)
    parser.add_argument("--expected-timeout-closeout-sha256", required=True)
    parser.add_argument("--permanent-timeout-closeout-registry", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    value = build(
        output=args.output,
        timeout_closeout_path=args.timeout_closeout,
        expected_timeout_closeout_sha256=(
            args.expected_timeout_closeout_sha256),
        permanent_timeout_closeout_registry=(
            args.permanent_timeout_closeout_registry))
    print(canonical({
        "status": value["status"],
        "content_sha256": value["content_sha256"],
        "scene_count": value["scene_count"],
        "ordered_identity_sha256": value["ordered_identity_sha256"],
    }))


if __name__ == "__main__":
    main()


__all__ = [
    "VERSION", "STATUS", "SEED_STARTS", "BATCH_MARKERS",
    "PER_FAMILY_PER_BATCH", "FAMILY_IDS", "BATCH_COUNT", "SCENE_COUNT",
    "contract", "producer_sources", "create_universe", "build",
    "read_saved_universe", "main",
]
