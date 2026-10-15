from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import zipfile

import numpy as np
import pytest

from backend.training import warehouse_r41_diagnostic_rcpd_v9_outer_split as subject
from backend.training.warehouse_native_common import canonical, digest, file_hash


def _fp(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def test_source_closure_contains_new_splitter_and_no_protected_reader():
    sources = subject.producer_sources()
    assert "backend/training/warehouse_r41_diagnostic_rcpd_v9_outer_split.py" in sources
    assert "backend/training/warehouse_r41_diagnostic_outer_failure_closeout_v9.py" in sources
    assert not [path for path in sources
                if "fresh_final_holdout" in path or "final_once" in path]


def _scene(label: str, seed: int, family: str, split: str) -> dict:
    return {
        "id": label,
        "seed": seed,
        "fingerprint": _fp(label),
        "family_id": family,
        "batch_index": seed % 3,
        "split": split,
        "snapshot": {"label": label},
    }


def _json(path: Path, value: dict) -> None:
    path.write_text(canonical(value) + "\n", encoding="utf-8")


def _npy(value: np.ndarray) -> bytes:
    stream = BytesIO()
    np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


def _rows(path: Path, fingerprints: list[str], *, private_byte: bytes = b"PRIVATE") -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name in sorted({field + ".npy" for field in subject.closeout_api.v8._ROW_FIELDS}):
            raw = (_npy(np.asarray(fingerprints, dtype="S64"))
                   if name == "scene_fingerprints.npy" else private_byte)
            archive.writestr(name, raw)


def _fixtures(tmp_path: Path, monkeypatch):
    families = subject.FAMILY_IDS
    monkeypatch.setattr(subject, "SOURCE_ROW_SCENE_COUNT", 12)
    monkeypatch.setattr(subject, "ORIGINAL_FIT_SCENE_COUNT", 6)
    monkeypatch.setattr(subject, "ORIGINAL_OUTER_SCENE_COUNT", 6)
    monkeypatch.setattr(subject, "CURRENT_FIT_SCENE_COUNT", 6)
    monkeypatch.setattr(subject, "CURRENT_OUTER_SCENE_COUNT", 6)
    monkeypatch.setattr(subject, "ORIGINAL_TRACE_IDENTITY_COUNT", 18)
    monkeypatch.setattr(subject, "PREVIOUS_DEVELOPMENT_SCENE_COUNT", 6)
    monkeypatch.setattr(subject, "FIXED_CANDIDATE_SCENE_COUNT", 30)
    monkeypatch.setattr(subject, "EXPECTED_REMAINING_SCENE_COUNT", 24)
    monkeypatch.setattr(subject, "EXPECTED_REMAINING_FAMILY_COUNTS",
                        {family: 4 for family in families})
    monkeypatch.setattr(subject, "FRESH_OUTER_SCENE_COUNT", 6)
    monkeypatch.setattr(subject, "FAMILY_QUOTAS", {family: 1 for family in families})
    monkeypatch.setattr(subject.retired_api, "EXPECTED_GLOBAL_EXPOSED_IDENTITIES", 6)
    monkeypatch.setattr(subject, "producer_sources", lambda: {"subject.py": "a" * 64})

    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    manifest = manifest_dir / "manifest.json"
    validation = manifest_dir / "validation.json"
    manifest.write_bytes(b"opaque manifest\n")
    validation.write_bytes(b"opaque validation\n")
    monkeypatch.setattr(subject, "EXPECTED_MANIFEST_SHA256", file_hash(manifest))
    monkeypatch.setattr(subject, "EXPECTED_MANIFEST_VALIDATION_SHA256", file_hash(validation))

    fit = [_scene(f"fit-{i}", 1_000 + i, families[i], "fit_supplement")
           for i in range(6)]
    old_outer = [_scene(f"old-{i}", 2_000 + i, families[i], "development_validation")
                 for i in range(6)]
    rejected = [_scene(f"rejected-{i}", 3_000 + i, families[i], "trace")
                for i in range(6)]
    original = {
        "version": subject.source_expansion_api.VERSION,
        "status": subject.source_expansion_api.STATUS,
        "fit_supplement": fit,
        "development_validation": old_outer,
        "selection_trace": {
            "fit_supplement": deepcopy(fit),
            "development_validation": [*deepcopy(old_outer), *deepcopy(rejected)],
        },
        "program_access": False,
        "program_predictions_access": False,
        "final_audit_rows_access": False,
        "final_labels_used_for_selection": False,
    }
    original["content_sha256"] = digest(original)
    original_path = tmp_path / "original.json"
    _json(original_path, original)
    monkeypatch.setattr(subject, "EXPECTED_ORIGINAL_EXPANSION_SHA256",
                        file_hash(original_path))

    current = [_scene(f"current-{i}", 4_000 + i, families[i], "development_validation")
               for i in range(6)]
    selected = [{key: row[key] for key in (
        "batch_index", "family_id", "seed", "fingerprint")} for row in current]
    current_registry = {
        "version": subject.old_outer_api.VERSION,
        "status": subject.old_outer_api.STATUS,
        "fit_supplement": deepcopy(fit),
        "development_validation": deepcopy(current),
        "selected_outer_identities": selected,
        "previously_exposed_validation_scene_fingerprints": sorted(
            row["fingerprint"] for row in old_outer),
        "program_access": False,
        "program_predictions_access": False,
        "final_audit_rows_access": False,
        "final_labels_used_for_selection": False,
        "formal_ready": False,
    }
    current_registry["content_sha256"] = digest(current_registry)
    current_path = tmp_path / "current.json"
    _json(current_path, current_registry)
    monkeypatch.setattr(subject, "EXPECTED_CURRENT_OUTER_REGISTRY_SHA256",
                        file_hash(current_path))
    current_report = {
        "version": subject.old_outer_api.REPORT_VERSION,
        "status": subject.old_outer_api.STATUS,
        "registry_file_sha256": file_hash(current_path),
        "registry_content_sha256": current_registry["content_sha256"],
        "selection": {"selected_identity_sha256": digest(selected)},
        "formal_ready": False,
    }
    current_report["content_sha256"] = digest(current_report)
    current_report_path = tmp_path / "current-report.json"
    _json(current_report_path, current_report)
    monkeypatch.setattr(subject, "EXPECTED_CURRENT_OUTER_REPORT_SHA256",
                        file_hash(current_report_path))

    rows_path = tmp_path / "failed-rows.npz"
    _rows(rows_path, [row["fingerprint"] for row in (*fit, *current)])
    monkeypatch.setattr(subject, "EXPECTED_FAILED_ROWS_SHA256", file_hash(rows_path))

    formal_rows = [_scene(f"formal-{i}", 5_000 + i, families[i], "formal")
                   for i in range(6)]
    formal = {
        "version": "warehouse-r41-diagnostic-conflict-dynamic-selection.v3",
        "release_eligible": True,
        "source_manifest_file_sha256": file_hash(manifest),
        "zero_action_overrides": True,
        "X": formal_rows[:3],
        "Y": formal_rows[3:],
    }
    formal_path = tmp_path / "formal.json"
    _json(formal_path, formal)
    monkeypatch.setattr(subject, "EXPECTED_FORMAL_SELECTION_SHA256",
                        file_hash(formal_path))

    previous_rows = [_scene(f"previous-{i}", 6_000 + i, families[i], "previous")
                     for i in range(6)]
    previous = {
        "version": "warehouse-r41-diagnostic-development-supplement.v1",
        "status": "passed",
        "program_access": False,
        "final_audit_rows_access": False,
        "scenes": previous_rows,
    }
    previous["content_sha256"] = digest(previous)
    previous_path = tmp_path / "previous.json"
    _json(previous_path, previous)
    monkeypatch.setattr(subject, "EXPECTED_PREVIOUS_DEVELOPMENT_SHA256",
                        file_hash(previous_path))

    retired_dir = tmp_path / "retired"
    retired_dir.mkdir()
    retired_rows = [{"seed": 7_000 + i, "fingerprint": _fp(f"retired-{i}")}
                    for i in range(6)]
    retired = {"exposed_identities": retired_rows}
    retired_path = retired_dir / "retired_identity_projection.json"
    _json(retired_path, retired)
    retired_report = retired_dir / "report.json"
    retired_report.write_bytes(b"opaque retired report\n")
    monkeypatch.setattr(subject, "EXPECTED_RETIRED_PROJECTION_SHA256",
                        file_hash(retired_path))
    monkeypatch.setattr(subject, "EXPECTED_RETIRED_REPORT_SHA256",
                        file_hash(retired_report))
    monkeypatch.setattr(subject.retired_api, "read_saved_projection",
                        lambda *args, **kwargs: deepcopy(retired))

    candidates = []
    fresh_seed = 10_000
    for family_index, family in enumerate(families):
        # One selection-trace identity per family is exposed.  Four remain.
        candidates.append({
            "batch_index": 0,
            "family_offset": 0,
            "family_id": family,
            "seed": rejected[family_index]["seed"],
            "fingerprint": rejected[family_index]["fingerprint"],
        })
        for offset in range(1, 5):
            candidates.append({
                "batch_index": offset % 3,
                "family_offset": offset,
                "family_id": family,
                "seed": fresh_seed,
                "fingerprint": _fp(f"fresh-{family}-{offset}"),
            })
            fresh_seed += 1
    monkeypatch.setattr(subject, "_candidate_identity_population",
                        lambda: deepcopy(candidates))

    projection = {
        "version": subject.closeout_api.VERSION + ".outer-observation-hash-projection.v1",
        "source_rows_file_sha256": file_hash(rows_path),
        "source_field": "authenticated observation_hashes",
        "split_field": "authenticated split_validation",
        "outer_row_count": 6,
        "unique_outer_observation_hash_count": 2,
        "outer_scene_count": 6,
        "outer_observation_hashes": [_fp("outer-obs-a"), _fp("outer-obs-b")],
        "outer_scene_fingerprints_sha256": digest(sorted(
            row["fingerprint"] for row in current)),
        "raw_observations_included": False,
        "actions_included": False,
        "probabilities_included": False,
        "labels_included": False,
    }
    projection["content_sha256"] = digest(projection)
    closeout = {
        "content_sha256": "c" * 64,
        "source": {
            "v8_combined_rows_sha256": file_hash(rows_path),
            "v8_outer_registry_sha256": file_hash(current_path),
            "v8_outer_registry_report_sha256": file_hash(current_report_path),
        },
        "consumed_outer": {
            "identities": selected,
            "observation_hash_projection": projection,
        },
        "disposition": {"eligible_for_outer_claim": False},
    }
    closeout_path = tmp_path / "failure_closeout.json"
    closeout_path.write_bytes(b"opaque closeout\n")
    closeout_sha = file_hash(closeout_path)
    monkeypatch.setattr(subject.closeout_api, "read_saved_closeout",
                        lambda *args, **kwargs: deepcopy(closeout))

    def materialize(identity, index):
        return {
            "id": f"v9-{index}",
            "split": "development_outer",
            "seed": identity["seed"],
            "fingerprint": identity["fingerprint"],
            "family_id": identity["family_id"],
            "batch_index": identity["batch_index"],
            "snapshot": {"index": index},
        }
    monkeypatch.setattr(subject, "_materialize_scene", materialize)

    paths = {
        "manifest_path": manifest,
        "failed_rows_path": rows_path,
        "original_expansion_path": original_path,
        "current_outer_registry_path": current_path,
        "current_outer_report_path": current_report_path,
        "failure_closeout_path": closeout_path,
        "expected_failure_closeout_sha256": closeout_sha,
        "permanent_closeout_registry": tmp_path,
        "formal_selection_path": formal_path,
        "previous_development_path": previous_path,
        "retired_projection_path": retired_path,
    }
    return paths, candidates, rejected, current, closeout


def test_source_closure_contains_both_v9_producers_and_no_protected_input_module():
    sources = subject.producer_sources()
    assert "backend/training/warehouse_r41_diagnostic_rcpd_v9_outer_split.py" in sources
    assert "backend/training/warehouse_r41_diagnostic_outer_failure_closeout_v9.py" in sources
    assert not [path for path in sources
                if "fresh_final_holdout" in path or "final_once" in path]


def test_fresh_v9_outer_excludes_all_exposed_identities_and_keeps_quotas(
        tmp_path, monkeypatch):
    paths, _, rejected, current, closeout = _fixtures(tmp_path, monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("forbidden Actor/observation/workload path")

    monkeypatch.setattr(subject.R41DiagnosticConflictWarehouseEnv,
                        "observations", forbidden)
    monkeypatch.setattr(subject.manifest_binding, "load_frozen_actor", forbidden)
    monkeypatch.setattr(subject.scenes_api, "load_frozen_actor", forbidden)
    monkeypatch.setattr(subject.scenes_api, "screen_scene", forbidden)
    monkeypatch.setattr(subject.scenes_api, "replay_workload_and_compare", forbidden)
    registry, report, projection = subject.create_registry(**paths)

    assert registry["status"] == report["status"] == subject.STATUS
    assert len(registry["development_outer"]) == 6
    assert set(registry["statistics"]["selected_outer_family_counts"].values()) == {1}
    selected = {row["fingerprint"] for row in registry["selected_outer_identities"]}
    assert not selected & {row["fingerprint"] for row in rejected}
    assert not selected & {row["fingerprint"] for row in current}
    assert registry["statistics"]["remaining_candidate_scene_count"] == 24
    assert registry["statistics"]["remaining_family_counts"] == {
        family: 4 for family in subject.FAMILY_IDS}
    assert registry["information_boundary"]["failed_rows_fields_read"] == [
        "scene_fingerprints"]
    assert registry["information_boundary"]["failed_rows_observations_read"] is False
    assert registry["information_boundary"]["failed_rows_actions_read"] is False
    assert registry["information_boundary"]["protected_final_access"] is False
    assert projection["outer_observation_hashes"] == closeout[
        "consumed_outer"]["observation_hash_projection"]["outer_observation_hashes"]
    assert projection["raw_observations_included"] is False
    assert projection["actions_included"] is False
    assert report["selection"]["salt"] == subject.SELECTION_SALT


def test_hostile_private_members_are_never_opened(tmp_path, monkeypatch):
    fingerprints = [_fp("one"), _fp("two")]
    path = tmp_path / "hostile.npz"
    _rows(path, fingerprints, private_byte=b"not-a-numpy-file-SENTINEL")
    monkeypatch.setattr(subject, "SOURCE_ROW_SCENE_COUNT", 2)
    _, actual = subject._scene_fingerprints_only(
        path, expected_sha256=file_hash(path))
    assert actual == set(fingerprints)


def test_private_payload_changes_cannot_change_selection(tmp_path, monkeypatch):
    paths, _, _, _, _ = _fixtures(tmp_path, monkeypatch)
    first, _, _ = subject.create_registry(**paths)
    with zipfile.ZipFile(paths["failed_rows_path"]) as archive:
        raw = archive.read("scene_fingerprints.npy")
    with zipfile.ZipFile(paths["failed_rows_path"], "w") as archive:
        for name in sorted({field + ".npy" for field in subject.closeout_api.v8._ROW_FIELDS}):
            archive.writestr(name, raw if name == "scene_fingerprints.npy"
                             else b"DIFFERENT-HOSTILE-PRIVATE-PAYLOAD")
    monkeypatch.setattr(subject, "EXPECTED_FAILED_ROWS_SHA256",
                        file_hash(paths["failed_rows_path"]))
    # The authenticated closeout must bind the byte-new archive for this
    # metamorphic test; its identity and hash projection remain unchanged.
    original_reader = subject.closeout_api.read_saved_closeout

    def rebound(*args, **kwargs):
        value = original_reader(*args, **kwargs)
        value["source"]["v8_combined_rows_sha256"] = file_hash(
            paths["failed_rows_path"])
        return value

    monkeypatch.setattr(subject.closeout_api, "read_saved_closeout", rebound)
    second, _, _ = subject.create_registry(**paths)
    assert first["selected_outer_identities"] == second["selected_outer_identities"]


def test_wrong_fixed_pool_remaining_count_fails_closed(tmp_path, monkeypatch):
    paths, candidates, _, _, _ = _fixtures(tmp_path, monkeypatch)
    candidates.pop()
    monkeypatch.setattr(subject, "_candidate_identity_population", lambda: candidates)
    monkeypatch.setattr(subject, "FIXED_CANDIDATE_SCENE_COUNT", 29)
    with pytest.raises(ValueError, match="remaining population differs"):
        subject.create_registry(**paths)


def test_closeout_identity_mismatch_fails_closed(tmp_path, monkeypatch):
    paths, _, _, _, _ = _fixtures(tmp_path, monkeypatch)
    original_reader = subject.closeout_api.read_saved_closeout

    def mismatch(*args, **kwargs):
        value = original_reader(*args, **kwargs)
        value["consumed_outer"]["identities"][0]["seed"] += 1
        return value

    monkeypatch.setattr(subject.closeout_api, "read_saved_closeout", mismatch)
    with pytest.raises(ValueError, match="does not bind"):
        subject.create_registry(**paths)


def test_build_writes_distinct_registry_projection_and_report(tmp_path, monkeypatch):
    paths, _, _, _, _ = _fixtures(tmp_path, monkeypatch)
    output = tmp_path / "v9-output"
    report = subject.build(**paths, output=output)
    assert sorted(path.name for path in output.iterdir()) == [
        "development_outer.json", "outer_observation_hashes.json", "report.json"]
    assert report["registry_file_sha256"] == file_hash(
        output / "development_outer.json")
    projection = (output / "outer_observation_hashes.json").read_text("utf-8")
    assert "PRIVATE" not in projection
    assert "observation_hashes" in projection
