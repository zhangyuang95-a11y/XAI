from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from backend.training import warehouse_r41_diagnostic_rcpd_v8_outer_split as subject
from backend.training.warehouse_native_common import canonical, digest, file_hash


def _fp(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def _scene(label: str, seed: int, family: str, split: str) -> dict:
    return {
        "id": label,
        "seed": seed,
        "fingerprint": _fp(label),
        "family_id": family,
        "split": split,
        "snapshot": {"label": label},
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(canonical(value) + "\n", encoding="utf-8")


def _fixtures(tmp_path: Path, monkeypatch):
    families = subject.FAMILY_IDS
    monkeypatch.setattr(subject, "FIT_SCENE_COUNT", 6)
    monkeypatch.setattr(subject, "OLD_OUTER_SCENE_COUNT", 6)
    monkeypatch.setattr(subject, "FRESH_OUTER_SCENE_COUNT", 6)
    monkeypatch.setattr(subject, "FIT_SUPPLEMENT_SCENES", 6)
    monkeypatch.setattr(subject, "VALIDATION_SCENES", 6)
    monkeypatch.setattr(subject, "SOURCE_ROW_SCENE_COUNT", 12)
    monkeypatch.setattr(subject, "FAMILY_QUOTAS", {family: 1 for family in families})
    monkeypatch.setattr(subject.retired_api, "EXPECTED_GLOBAL_EXPOSED_IDENTITIES", 6)
    monkeypatch.setattr(subject, "producer_sources", lambda: {"subject.py": "a" * 64})

    manifest_dir = tmp_path / "manifest"; manifest_dir.mkdir()
    manifest = manifest_dir / "manifest.json"; manifest.write_bytes(b"opaque manifest\n")
    validation = manifest_dir / "validation.json"; validation.write_bytes(b"opaque validation\n")
    monkeypatch.setattr(subject, "EXPECTED_MANIFEST_SHA256", file_hash(manifest))
    monkeypatch.setattr(subject, "EXPECTED_MANIFEST_VALIDATION_SHA256", file_hash(validation))

    fit = [_scene(f"fit-{i}", 1_000 + i, families[i], "fit_supplement") for i in range(6)]
    old_outer = [_scene(f"old-outer-{i}", 2_000 + i, families[i],
                        "development_validation") for i in range(6)]
    rejected = [_scene(f"screened-reject-{i}", 2_100 + i, families[i],
                       "play_candidates") for i in range(6)]
    expansion = {
        "version": subject.expansion_api.VERSION,
        "status": subject.expansion_api.STATUS,
        "fit_supplement": fit,
        "development_validation": old_outer,
        "selection_trace": {
            "development_validation": [
                {**deepcopy(row), "accepted": True} for row in old_outer
            ] + [{**deepcopy(row), "accepted": False} for row in rejected],
            "fit_supplement": [{**deepcopy(row), "accepted": True} for row in fit],
        },
        "program_access": False,
        "program_predictions_access": False,
        "final_audit_rows_access": False,
        "final_labels_used_for_selection": False,
        "bindings": {
            "actor_sha256": subject.manifest_binding.FROZEN_ACTOR_SHA256,
            "actor_parameters_sha256": (
                subject.manifest_binding.EXPECTED_ACTOR_PARAMETERS_SHA256),
            "source_manifest_file_sha256": file_hash(manifest),
            "designation_file_sha256": (
                subject.expansion_api.EXPECTED_DESIGNATION_SHA256),
            "previous_development_file_sha256": "0" * 64,
        },
    }
    expansion["content_sha256"] = digest(expansion)
    expansion_path = tmp_path / "development_expansion.json"; _write_json(expansion_path, expansion)

    source_scene_fingerprints = np.asarray(
        [row["fingerprint"] for row in (*fit, *old_outer)], dtype="S64")
    rows = tmp_path / "rows.npz"
    np.savez_compressed(
        rows, scene_fingerprints=source_scene_fingerprints,
        observations=np.arange(24, dtype=np.float32).reshape(12, 2),
        probabilities=np.zeros((12, 5), dtype=np.float32),
        action_indices=np.zeros(12, dtype=np.uint8))
    monkeypatch.setattr(subject, "EXPECTED_SOURCE_ROWS_SHA256", file_hash(rows))

    formal_rows = [_scene(f"formal-{i}", 3_000 + i, families[i], "formal") for i in range(6)]
    formal = {
        "version": "warehouse-r41-diagnostic-conflict-dynamic-selection.v3",
        "release_eligible": True,
        "source_manifest_file_sha256": file_hash(manifest),
        "zero_action_overrides": True,
        "X": formal_rows[:3], "Y": formal_rows[3:],
    }
    formal_path = tmp_path / "formal.json"; _write_json(formal_path, formal)
    monkeypatch.setattr(subject, "EXPECTED_FORMAL_SELECTION_SHA256", file_hash(formal_path))

    previous_rows = [
        _scene(f"previous-{i}", 4_000 + i, families[i % 6], "previous")
        for i in range(64)
    ]
    previous = {
        "version": "warehouse-r41-diagnostic-development-supplement.v1",
        "status": "passed", "program_access": False,
        "final_audit_rows_access": False, "scenes": previous_rows,
    }
    previous["content_sha256"] = digest(previous)
    previous_path = tmp_path / "previous.json"; _write_json(previous_path, previous)
    monkeypatch.setattr(subject, "EXPECTED_PREVIOUS_DEVELOPMENT_SHA256", file_hash(previous_path))
    expansion["bindings"]["previous_development_file_sha256"] = file_hash(previous_path)
    expansion["content_sha256"] = digest({
        key: value for key, value in expansion.items() if key != "content_sha256"})
    _write_json(expansion_path, expansion)
    monkeypatch.setattr(subject, "EXPECTED_SOURCE_EXPANSION_SHA256", file_hash(expansion_path))

    retired_dir = tmp_path / "retired"; retired_dir.mkdir()
    retired_rows = [
        {"seed": 5_000 + i, "fingerprint": _fp(f"retired-{i}")}
        for i in range(6)
    ]
    retired = {"exposed_identities": retired_rows}
    retired_path = retired_dir / "projection.json"; _write_json(retired_path, retired)
    retired_report = retired_dir / "report.json"; retired_report.write_bytes(b"opaque report\n")
    monkeypatch.setattr(subject, "EXPECTED_RETIRED_PROJECTION_SHA256", file_hash(retired_path))
    monkeypatch.setattr(subject, "EXPECTED_RETIRED_REPORT_SHA256", file_hash(retired_report))
    monkeypatch.setattr(subject.retired_api, "read_saved_projection",
                        lambda *args, **kwargs: deepcopy(retired))

    candidates = []
    seed = 10_000
    # First candidate in every family is trace-touched; sufficient untouched
    # candidates remain and must be selected deterministically.
    for family_index, family in enumerate(families):
        for offset in range(5):
            if offset == 0:
                row = rejected[family_index]
                candidates.append({
                    "batch_index": 0, "family_offset": offset,
                    "family_id": family, "seed": row["seed"],
                    "fingerprint": row["fingerprint"],
                })
            else:
                candidates.append({
                    "batch_index": offset % 3, "family_offset": offset,
                    "family_id": family, "seed": seed,
                    "fingerprint": _fp(f"fresh-{family}-{offset}"),
                })
                seed += 1
    monkeypatch.setattr(subject, "_candidate_identity_population",
                        lambda: deepcopy(candidates))

    def materialize(identity, index):
        return {
            "id": f"fresh-outer-{index}", "split": "development_validation",
            "seed": identity["seed"], "fingerprint": identity["fingerprint"],
            "family_id": identity["family_id"],
            "batch_index": identity["batch_index"], "snapshot": {"index": index},
        }
    monkeypatch.setattr(subject, "_materialize_scene", materialize)
    return {
        "manifest_path": manifest, "source_rows_path": rows,
        "source_expansion_path": expansion_path,
        "formal_selection_path": formal_path,
        "previous_development_path": previous_path,
        "retired_projection_path": retired_path,
    }, fit, old_outer, rejected


def test_fresh_outer_never_relabels_candidate_v3_or_trace_rows(tmp_path, monkeypatch):
    paths, fit, old_outer, rejected = _fixtures(tmp_path, monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("forbidden Actor/observation/workload path")

    monkeypatch.setattr(subject.R41DiagnosticConflictWarehouseEnv,
                        "observations", forbidden)
    monkeypatch.setattr(subject.manifest_binding, "load_frozen_actor", forbidden)
    monkeypatch.setattr(subject.scenes_api, "load_frozen_actor", forbidden)
    monkeypatch.setattr(subject.scenes_api, "screen_scene", forbidden)
    monkeypatch.setattr(subject.scenes_api, "replay_workload_and_compare", forbidden)
    registry, report = subject.create_registry(**paths)

    assert registry["status"] == subject.STATUS
    assert report["version"] == registry["version"] == subject.VERSION
    assert subject.FIT_SUPPLEMENT_SCENES == subject.FIT_SCENE_COUNT
    assert subject.VALIDATION_SCENES == subject.FRESH_OUTER_SCENE_COUNT
    assert registry["fit_supplement"] == fit
    assert len(registry["development_validation"]) == 6
    assert set(registry["statistics"]["fresh_outer_family_counts"].values()) == {1}
    fresh = {row["fingerprint"] for row in registry["development_validation"]}
    old = {row["fingerprint"] for row in old_outer}
    touched = {row["fingerprint"] for row in rejected}
    source = set(np.char.decode(np.load(paths["source_rows_path"])["scene_fingerprints"], "ascii"))
    assert not fresh & source
    assert not fresh & old
    assert not fresh & touched
    assert registry["information_boundary"]["outer_actor_rows_collected"] is False
    assert registry["information_boundary"]["actor_loaded_or_inferred"] is False
    assert registry["information_boundary"]["observations_generated"] is False
    assert registry["information_boundary"][
        "candidate_population_disjoint_from_all_manifest_base_splits_authenticated"
    ] is True
    assert registry["statistics"][
        "fresh_outer_manifest_registered_scene_overlap"] == 0
    assert registry["bindings"]["actor_sha256"] == (
        subject.manifest_binding.FROZEN_ACTOR_SHA256)
    assert registry["bindings"]["actor_parameters_sha256"] == (
        subject.manifest_binding.EXPECTED_ACTOR_PARAMETERS_SHA256)
    assert registry["bindings"]["source_manifest_file_sha256"] == file_hash(
        paths["manifest_path"])
    assert registry["bindings"]["designation_file_sha256"] == (
        subject.expansion_api.EXPECTED_DESIGNATION_SHA256)
    assert report["selection"]["selected_identity_sha256"] == digest(
        registry["selected_outer_identities"])


def test_action_probability_and_observation_payload_cannot_change_selection(tmp_path, monkeypatch):
    paths, _, _, _ = _fixtures(tmp_path, monkeypatch)
    first, _ = subject.create_registry(**paths)
    with np.load(paths["source_rows_path"], allow_pickle=False) as archive:
        scenes = archive["scene_fingerprints"].copy()
    np.savez_compressed(
        paths["source_rows_path"], scene_fingerprints=scenes,
        observations=np.full((12, 900), 999, dtype=np.float32),
        probabilities=np.eye(5, dtype=np.float32), action_indices=np.arange(12))
    monkeypatch.setattr(subject, "EXPECTED_SOURCE_ROWS_SHA256",
                        file_hash(paths["source_rows_path"]))
    second, _ = subject.create_registry(**paths)
    assert first["selected_outer_identities"] == second["selected_outer_identities"]


def test_npz_reader_opens_only_scene_identity_member(tmp_path, monkeypatch):
    fingerprints = np.asarray([_fp("one"), _fp("two")], dtype="S64")
    encoded = BytesIO(); np.save(encoded, fingerprints, allow_pickle=False)
    path = tmp_path / "hostile.npz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("scene_fingerprints.npy", encoded.getvalue())
        archive.writestr("observations.npy", b"not-a-numpy-file-SENTINEL")
        archive.writestr("probabilities.npy", b"private-SENTINEL")
    monkeypatch.setattr(subject, "SOURCE_ROW_SCENE_COUNT", 2)
    _, result = subject._scene_fingerprints_only(
        path, expected_sha256=file_hash(path))
    assert result == set(map(str, fingerprints.astype(str)))


def test_actual_identity_materialisation_never_calls_forbidden_paths(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("forbidden Actor/observation/workload path")

    monkeypatch.setattr(subject.R41DiagnosticConflictWarehouseEnv, "reset", forbidden)
    monkeypatch.setattr(subject.R41DiagnosticConflictWarehouseEnv, "observations", forbidden)
    monkeypatch.setattr(subject.manifest_binding, "load_frozen_actor", forbidden)
    monkeypatch.setattr(subject.scenes_api, "load_frozen_actor", forbidden)
    monkeypatch.setattr(subject.scenes_api, "screen_scene", forbidden)
    monkeypatch.setattr(subject.scenes_api, "replay_workload_and_compare", forbidden)

    identities, _ = subject.retired_api._candidate_identity_batch(0, per_family=1)
    identity = {
        **identities[0], "batch_index": 0, "family_offset": 0,
        "family_id": subject.FAMILY_IDS[0],
    }
    scene = subject._materialize_scene(identity, 0)
    assert scene["fingerprint"] == identity["fingerprint"]
    assert scene["family_id"] == identity["family_id"]


def test_insufficient_unexposed_family_fails_closed(tmp_path, monkeypatch):
    paths, _, _, rejected = _fixtures(tmp_path, monkeypatch)
    candidates = subject._candidate_identity_population()
    blocked_family = subject.FAMILY_IDS[0]
    for row in candidates:
        if row["family_id"] == blocked_family:
            row["seed"] = rejected[0]["seed"]
            row["fingerprint"] = rejected[0]["fingerprint"]
    monkeypatch.setattr(subject, "_candidate_identity_population", lambda: candidates)
    with pytest.raises(ValueError, match="Insufficient fresh candidate"):
        subject.create_registry(**paths)
