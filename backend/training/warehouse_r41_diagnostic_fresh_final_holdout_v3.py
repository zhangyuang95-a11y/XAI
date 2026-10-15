"""Freeze a fresh explanation holdout without reading an explanation program.

The source manifest's play-candidate pool was generated before RCPD fitting.
This producer deterministically orders that pool, excludes every registered and
participant scene plus both retired fresh holdouts, replays each retired
final-audit workload with its original scene indexes, and admits the first
public-observation-disjoint scenes that fill the six family quotas.  It never
loads a program or program prediction.  The previously audited final split is
treated as burned and is included in the exclusion observation set.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.training.warehouse_native_evaluation import critical_groups
from backend.training.warehouse_r41_diagnostic_conflict_scenarios import (
    FAMILY_IDS, VERSION as MANIFEST_VERSION, validate_diagnostic_manifest,
)
from backend.training.warehouse_r41_diagnostic_workload_screen import (
    CONTRACT_SHA256 as WORKLOAD_CONTRACT_SHA256,
    VERSION as WORKLOAD_VERSION,
    screen_scene,
)
from backend.warehouse_r41_diagnostic_online_runtime import (
    R41DiagnosticOnlineAlignmentRuntime,
)
from env.warehouse.navigation import ACTIONS
from env.warehouse_native.partners import partner_action


VERSION = "warehouse-r41-diagnostic-fresh-final-holdout.v3"
PARTNERS = ("skilled", "assertive", "noisy")
TOTAL_SCENES = 64
FAMILY_QUOTAS = dict(zip(FAMILY_IDS, (11, 11, 11, 11, 10, 10)))
ORDER_SALT = "r41-diagnostic-fresh-final-after-burned-v4-v5-and-v6-audits-20260912"
MAX_JSON_BYTES = 512 * 1024 * 1024
ROOT = Path(__file__).resolve().parents[2]


def contract() -> dict[str, Any]:
    return {
        "version": VERSION,
        "source_manifest_version": MANIFEST_VERSION,
        "workload_screen_version": WORKLOAD_VERSION,
        "workload_screen_contract_sha256": WORKLOAD_CONTRACT_SHA256,
        "purpose": "one frozen explanation final audit after v4, v5 and v6 final retirement",
        "selection_population": "source manifest play_candidates only",
        "selection_order": "ascending sha256(order salt, scene fingerprint)",
        "selection_rule": (
            "first exact-final-workload-safe scene in each deterministic family "
            "stream whose program-input public observations do not overlap RCPD "
            "development, the retired final, or an already accepted fresh scene"
        ),
        "family_quotas": deepcopy(FAMILY_QUOTAS),
        "scene_count": TOTAL_SCENES,
        "partners": list(PARTNERS),
        "horizon": 120,
        "program_access": False,
        "program_predictions_access": False,
        "participant_data_access": False,
        "actor_selection": False,
        "final_labels_used_for_selection": False,
        "retired_final_reuse": False,
        "previous_fresh_final_reuse": False,
        "formal_ready": False,
    }


def producer_sources() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "backend/training/warehouse_r41_diagnostic_workload_screen.py",
        ROOT / "backend/training/warehouse_r41_diagnostic_conflict_scenarios.py",
        ROOT / "backend/warehouse_r41_diagnostic_online_runtime.py",
    )
    return {str(path.relative_to(ROOT)): file_hash(path) for path in paths}


def _read(path: Path) -> dict[str, Any]:
    if (not path.is_file() or path.is_symlink() or path.resolve() != path.absolute()
            or path.stat().st_size > MAX_JSON_BYTES):
        raise ValueError("Fresh-final JSON input is missing, linked, or oversized")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Fresh-final JSON input must be an object")
    return value


def _write(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink() or path.parent.is_symlink():
        raise ValueError("Fresh-final output path is unsafe")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                         | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush(); os.fsync(stream.fileno())


def _obs_hash(observation: np.ndarray) -> str:
    value = np.asarray(observation, dtype="<f4")
    if value.shape != (197,) or not np.isfinite(value).all():
        raise ValueError("Fresh-final public observation differs")
    return sha256(value.tobytes(order="C")).hexdigest()


def _exact_workload_observation_hashes(
    runtime: R41DiagnosticOnlineAlignmentRuntime,
    scene: Mapping[str, Any], scene_index: int,
) -> set[str]:
    """Collect every public observation that the final audit may give a program."""
    result: set[str] = set()
    for partner_index, partner in enumerate(PARTNERS):
        env = runtime.environment(scene)
        rng = np.random.default_rng(17000 + partner_index * 1000 + scene_index)
        first_after = None
        while not env.done:
            source = env.snapshot()
            result.add(_obs_hash(env.observations()["robot_2"]))
            groups = critical_groups(env, "robot_2")
            if env.state.frame % 10 == 0 and groups:
                for action in ACTIONS:
                    branch = runtime.from_snapshot(source)
                    transition = runtime.step(branch, action)
                    if not transition["done"]:
                        result.add(_obs_hash(branch.observations()["robot_2"]))
            player = partner_action(env, "robot_1", partner, rng)
            transition = runtime.step(env, player)
            if first_after is None:
                first_after = transition["after"]
        # The language matrix may execute the documented WAIT-three branch
        # from the first skilled transition.  Include it conservatively for
        # every partner so selection cannot exploit the language subset.
        if first_after is not None:
            branch = runtime.from_snapshot(first_after)
            for _ in range(3):
                if branch.done:
                    break
                result.add(_obs_hash(branch.observations()["robot_2"]))
                runtime.step(branch, "WAIT")
    return result


def _development_hashes(rows_path: Path) -> set[str]:
    if (not rows_path.is_file() or rows_path.is_symlink()
            or rows_path.resolve() != rows_path.absolute()):
        raise ValueError("Fresh-final RCPD rows are unsafe")
    with np.load(rows_path, allow_pickle=False) as rows:
        if "observation_hashes" not in rows.files:
            raise ValueError("Fresh-final RCPD observation hashes are missing")
        values = np.char.decode(rows["observation_hashes"], "ascii")
    if any(len(value) != 64 for value in values):
        raise ValueError("Fresh-final RCPD observation hash differs")
    return set(map(str, values))


def _ordered_candidates(manifest: Mapping[str, Any], family: str) -> list[dict]:
    rows = [deepcopy(row) for batch in manifest["candidate_batches"]
            for row in batch if row["family_id"] == family]
    rows.sort(key=lambda row: digest({"salt": ORDER_SALT,
                                     "fingerprint": row["fingerprint"]}))
    return rows


def _retired_scenes(path: Path, expected_version: str) -> list[dict]:
    value = _read(path)
    if (value.get("version") != expected_version
            or value.get("program_access") is not False
            or value.get("statistics", {}).get("accepted") != TOTAL_SCENES
            or value.get("statistics", {}).get("public_observation_overlap") != 0
            or value.get("content_sha256") != digest({
                key: item for key, item in value.items() if key != "content_sha256"
            })):
        raise ValueError("Retired fresh-final holdout identity differs")
    scenes = value.get("scenes")
    if (not isinstance(scenes, list) or len(scenes) != TOTAL_SCENES
            or len({row.get("fingerprint") for row in scenes}) != TOTAL_SCENES):
        raise ValueError("Retired fresh-final scene registry differs")
    return deepcopy(scenes)


def _select(*, runtime: R41DiagnosticOnlineAlignmentRuntime,
            actor, manifest: Mapping[str, Any], selected: Mapping[str, Any],
            development_hashes: set[str],
            previous_fresh_scene_sets: Sequence[Sequence[Mapping[str, Any]]],
            ) -> tuple[list[dict], list[dict], dict]:
    base_fingerprints = {row["fingerprint"] for rows in manifest["splits"].values()
                         for row in rows}
    participant_fingerprints = {row["fingerprint"] for key in ("X", "Y")
                                for row in selected[key]}
    retired_hashes: set[str] = set()
    for index, scene in enumerate(manifest["splits"]["final_test"]):
        retired_hashes.update(_exact_workload_observation_hashes(runtime, scene, index))
    previous_fresh_hashes: set[str] = set()
    for scene_set in previous_fresh_scene_sets:
        for index, scene in enumerate(scene_set):
            previous_fresh_hashes.update(
                _exact_workload_observation_hashes(runtime, scene, index))
    forbidden = set(development_hashes) | retired_hashes | previous_fresh_hashes
    accepted: list[dict] = []
    trace: list[dict] = []
    accepted_hashes: set[str] = set()
    for family in FAMILY_IDS:
        family_count = 0
        for scene in _ordered_candidates(manifest, family):
            if family_count >= FAMILY_QUOTAS[family]:
                break
            reason = None
            if scene["fingerprint"] in base_fingerprints | participant_fingerprints:
                reason = "registered_scene_overlap"
                receipt = None
                hashes: set[str] = set()
            else:
                index = len(accepted)
                receipt = screen_scene(scene, split="final_test",
                                       scene_index=index, actor=actor)
                if receipt["passed"] is not True:
                    reason = "exact_workload_failed"
                    hashes = set()
                else:
                    hashes = _exact_workload_observation_hashes(runtime, scene, index)
                    if hashes & forbidden:
                        reason = "development_or_retired_observation_overlap"
                    elif hashes & accepted_hashes:
                        reason = "fresh_holdout_observation_overlap"
            admitted = reason is None
            trace.append({"family_id": family, "fingerprint": scene["fingerprint"],
                          "scene_index": len(accepted), "accepted": admitted,
                          "rejection_reason": reason,
                          "workload_receipt": receipt,
                          "public_observation_hash_count": len(hashes),
                          "public_observation_hashes_sha256": digest(sorted(hashes))})
            if admitted:
                frozen = deepcopy(scene)
                frozen["id"] = f"diagnostic_fresh_final_{len(accepted):04d}"
                frozen["split"] = "fresh_final_test"
                frozen["workload_screen"] = receipt
                accepted.append(frozen)
                accepted_hashes.update(hashes)
                forbidden.update(hashes)
                family_count += 1
        if family_count != FAMILY_QUOTAS[family]:
            raise RuntimeError(f"Fresh-final family quota unavailable: {family}")
    stats = {
        "accepted": len(accepted),
        "evaluated": len(trace),
        "rejected": Counter(row["rejection_reason"] for row in trace
                            if row["rejection_reason"]),
        "families": Counter(row["family_id"] for row in accepted),
        "development_public_observation_count": len(development_hashes),
        "retired_final_public_observation_count": len(retired_hashes),
        "previous_fresh_final_public_observation_count": len(previous_fresh_hashes),
        "fresh_final_public_observation_count": len(accepted_hashes),
        "public_observation_overlap": 0,
    }
    stats["rejected"] = dict(sorted(stats["rejected"].items()))
    stats["families"] = dict(sorted(stats["families"].items()))
    return accepted, trace, stats


def build(*, actor_path: str | Path, protocol_path: str | Path,
          manifest_path: str | Path, selected_scenes_path: str | Path,
          rcpd_rows_path: str | Path, rcpd_report_path: str | Path,
          retired_holdout_paths: Sequence[str | Path],
          output: str | Path) -> dict[str, Any]:
    base_paths = [Path(value).expanduser().absolute() for value in (
        actor_path, protocol_path, manifest_path, selected_scenes_path,
        rcpd_rows_path, rcpd_report_path)]
    retired_paths = [Path(value).expanduser().absolute()
                     for value in retired_holdout_paths]
    paths = [*base_paths, *retired_paths]
    if len(retired_paths) != 2:
        raise ValueError("Fresh-final v3 requires exactly two retired holdouts")
    if any(not path.is_file() or path.is_symlink() or path.resolve() != path
           for path in paths):
        raise ValueError("Fresh-final input must be a canonical regular file")
    (actor_path, protocol_path, manifest_path, selected_path, rows_path,
     report_path) = base_paths
    manifest = _read(manifest_path)
    validate_diagnostic_manifest(manifest, replay=False)
    selected = _read(selected_path)
    report = _read(report_path)
    if (selected.get("release_eligible") is not True
            or selected.get("actor_sha256") != manifest["frozen_actor"]["sha256"]
            or report.get("version") != "warehouse-r41-diagnostic-rcpd.v7"
            or report.get("status") != "passed"
            or report.get("program_file_sha256") is None
            or report.get("evidence_artifacts", {}).get("rows.npz") != file_hash(rows_path)):
        raise ValueError("Fresh-final frozen development inputs differ")
    protocol = _read(protocol_path)
    runtime = R41DiagnosticOnlineAlignmentRuntime(
        actor_path, training_protocol_path=protocol_path, manifest_path=manifest_path,
        expected_actor_sha256=file_hash(actor_path),
        expected_training_protocol_file_sha256=file_hash(protocol_path),
        expected_training_protocol_content_sha256=digest(protocol),
        expected_manifest_file_sha256=file_hash(manifest_path),
        expected_manifest_content_sha256=manifest["content_sha256"],
        expected_manifest_semantic_sha256=digest(manifest))
    actor = runtime.actor
    development = _development_hashes(rows_path)
    scenes, trace, statistics = _select(runtime=runtime, actor=actor,
        manifest=manifest, selected=selected, development_hashes=development,
        previous_fresh_scene_sets=[
            _retired_scenes(retired_paths[0],
                "warehouse-r41-diagnostic-fresh-final-holdout.v1"),
            _retired_scenes(retired_paths[1],
                "warehouse-r41-diagnostic-fresh-final-holdout.v2"),
        ])
    bindings = {
        "actor_sha256": file_hash(actor_path),
        "protocol_sha256": file_hash(protocol_path),
        "source_manifest_sha256": file_hash(manifest_path),
        "source_manifest_content_sha256": manifest["content_sha256"],
        "selected_scenes_sha256": file_hash(selected_path),
        "rcpd_rows_sha256": file_hash(rows_path),
        "rcpd_report_sha256": file_hash(report_path),
        "retired_holdout_sha256": [file_hash(path) for path in retired_paths],
        "contract_sha256": digest(contract()),
        "producer_sources_sha256": digest(producer_sources()),
    }
    holdout = {"version": VERSION, "contract": contract(), "bindings": bindings,
        "scenes": scenes, "selection_trace": trace, "statistics": statistics,
        "program_access": False, "formal_ready": False}
    holdout["content_sha256"] = digest(holdout)
    output = Path(output).expanduser().absolute()
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    _write(output / "holdout.json", holdout)
    summary = {"version": VERSION, "status": "passed", "bindings": bindings,
        "holdout_file_sha256": file_hash(output / "holdout.json"),
        "holdout_content_sha256": holdout["content_sha256"],
        "statistics": statistics, "producer_sources": producer_sources(),
        "program_access": False, "formal_ready": False}
    _write(output / "report.json", summary)
    read_saved_holdout(output,
        expected_holdout_sha256=file_hash(output / "holdout.json"),
        expected_report_sha256=file_hash(output / "report.json"),
        actor_path=actor_path, protocol_path=protocol_path,
        manifest_path=manifest_path, selected_scenes_path=selected_path,
        rcpd_rows_path=rows_path, rcpd_report_path=report_path,
        retired_holdout_paths=retired_paths)
    return summary


def read_saved_holdout(output: str | Path, *, expected_holdout_sha256: str,
                       expected_report_sha256: str,
                       actor_path: str | Path, protocol_path: str | Path,
                       manifest_path: str | Path,
                       selected_scenes_path: str | Path,
                       rcpd_rows_path: str | Path,
                       rcpd_report_path: str | Path,
                       retired_holdout_paths: Sequence[str | Path]) -> dict[str, Any]:
    """Regenerate the deterministic selection and replay every considered scene."""
    output = Path(output).expanduser().absolute()
    holdout_path, saved_report_path = output / "holdout.json", output / "report.json"
    if (not output.is_dir() or output.is_symlink() or output.resolve() != output
            or file_hash(holdout_path) != expected_holdout_sha256
            or file_hash(saved_report_path) != expected_report_sha256):
        raise ValueError("Fresh-final saved evidence hash differs")
    base_paths = [Path(value).expanduser().absolute() for value in (
        actor_path, protocol_path, manifest_path, selected_scenes_path,
        rcpd_rows_path, rcpd_report_path)]
    retired_paths = [Path(value).expanduser().absolute()
                     for value in retired_holdout_paths]
    paths = [*base_paths, *retired_paths]
    if len(retired_paths) != 2:
        raise ValueError("Fresh-final v3 replay requires two retired holdouts")
    if any(not path.is_file() or path.is_symlink() or path.resolve() != path
           for path in paths):
        raise ValueError("Fresh-final replay input must be canonical")
    (actor_path, protocol_path, manifest_path, selected_path, rows_path,
     report_path) = base_paths
    manifest = _read(manifest_path)
    validate_diagnostic_manifest(manifest, replay=False)
    selected = _read(selected_path)
    rcpd_report = _read(report_path)
    if (selected.get("release_eligible") is not True
            or selected.get("actor_sha256") != manifest["frozen_actor"]["sha256"]
            or rcpd_report.get("version") != "warehouse-r41-diagnostic-rcpd.v7"
            or rcpd_report.get("status") != "passed"
            or rcpd_report.get("evidence_artifacts", {}).get("rows.npz")
                != file_hash(rows_path)):
        raise ValueError("Fresh-final replay inputs differ")
    protocol = _read(protocol_path)
    runtime = R41DiagnosticOnlineAlignmentRuntime(
        actor_path, training_protocol_path=protocol_path, manifest_path=manifest_path,
        expected_actor_sha256=file_hash(actor_path),
        expected_training_protocol_file_sha256=file_hash(protocol_path),
        expected_training_protocol_content_sha256=digest(protocol),
        expected_manifest_file_sha256=file_hash(manifest_path),
        expected_manifest_content_sha256=manifest["content_sha256"],
        expected_manifest_semantic_sha256=digest(manifest))
    scenes, trace, statistics = _select(runtime=runtime, actor=runtime.actor,
        manifest=manifest, selected=selected,
        development_hashes=_development_hashes(rows_path),
        previous_fresh_scene_sets=[
            _retired_scenes(retired_paths[0],
                "warehouse-r41-diagnostic-fresh-final-holdout.v1"),
            _retired_scenes(retired_paths[1],
                "warehouse-r41-diagnostic-fresh-final-holdout.v2"),
        ])
    bindings = {
        "actor_sha256": file_hash(actor_path),
        "protocol_sha256": file_hash(protocol_path),
        "source_manifest_sha256": file_hash(manifest_path),
        "source_manifest_content_sha256": manifest["content_sha256"],
        "selected_scenes_sha256": file_hash(selected_path),
        "rcpd_rows_sha256": file_hash(rows_path),
        "rcpd_report_sha256": file_hash(report_path),
        "retired_holdout_sha256": [file_hash(path) for path in retired_paths],
        "contract_sha256": digest(contract()),
        "producer_sources_sha256": digest(producer_sources()),
    }
    expected_holdout = {"version": VERSION, "contract": contract(),
        "bindings": bindings, "scenes": scenes, "selection_trace": trace,
        "statistics": statistics, "program_access": False, "formal_ready": False}
    expected_holdout["content_sha256"] = digest(expected_holdout)
    expected_report = {"version": VERSION, "status": "passed", "bindings": bindings,
        "holdout_file_sha256": expected_holdout_sha256,
        "holdout_content_sha256": expected_holdout["content_sha256"],
        "statistics": statistics, "producer_sources": producer_sources(),
        "program_access": False, "formal_ready": False}
    if (_read(holdout_path) != expected_holdout
            or _read(saved_report_path) != expected_report):
        raise ValueError("Fresh-final selection differs from independent replay")
    return expected_report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--selected-scenes", required=True)
    parser.add_argument("--rcpd-rows", required=True)
    parser.add_argument("--rcpd-report", required=True)
    parser.add_argument("--retired-holdout", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = build(actor_path=args.actor, protocol_path=args.protocol,
        manifest_path=args.manifest, selected_scenes_path=args.selected_scenes,
        rcpd_rows_path=args.rcpd_rows, rcpd_report_path=args.rcpd_report,
        retired_holdout_paths=args.retired_holdout,
        output=args.output)
    print(canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["VERSION", "TOTAL_SCENES", "FAMILY_QUOTAS", "contract",
           "producer_sources", "build", "read_saved_holdout", "main"]
