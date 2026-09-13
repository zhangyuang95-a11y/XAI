from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import inspect
import io
from pathlib import Path
import zipfile

import numpy as np
import pytest

from backend.training import warehouse_r41_diagnostic_outer_collection_v9 as subject
from backend.training import warehouse_r41_diagnostic_outer_hash_projection_v9 as projection_api
from backend.training.warehouse_native_common import canonical, digest, file_hash


def _arrays(count: int = 3) -> dict[str, np.ndarray]:
    observations = np.arange(count * 197, dtype=np.float32).reshape(count, 197)
    hashes = [subject.rows_v7.legacy._obs_hash(row) for row in observations]
    return {
        "observations": observations,
        "probabilities": np.tile(
            np.asarray([[.1, .2, .3, .15, .25]], dtype=np.float32), (count, 1)),
        "action_indices": np.asarray([2] * count, dtype=np.uint8),
        "weights": np.ones(count, dtype=np.float32),
        "observation_hashes": np.asarray(hashes, dtype="S64"),
        "scene_fingerprints": np.asarray([digest({"scene": 1})] * count, dtype="S64"),
        "episode_ids": np.asarray(["episode"] * count, dtype="S180"),
        "frames": np.arange(count, dtype=np.int16),
        "group_bits": np.asarray([0, 1, 1][:count], dtype=np.uint8),
        "kinds": np.asarray(["ordinary", "intervention", "intervention"][:count],
                            dtype="S16"),
        "anchor_ids": np.asarray(["", "a", "a"][:count], dtype="S240"),
        "branch_actions": np.asarray(["", "WAIT", "UP"][:count], dtype="S8"),
        "physical_hashes": np.asarray(
            ["", digest({"physical": 1}), digest({"physical": 2})][:count],
            dtype="S64"),
        "source_state_hashes": np.asarray(
            [digest({"state": index}) for index in range(count)], dtype="S64"),
        "submitted_equal": np.ones(count, dtype=np.bool_),
        "trajectory_done": np.asarray([False, False, True][:count], dtype=np.bool_),
        "split_validation": np.ones(count, dtype=np.bool_),
    }


def _projection(arrays: dict[str, np.ndarray]) -> dict:
    return projection_api.projection_from_arrays(
        arrays, registry_file_sha256="1" * 64,
        registry_content_sha256="2" * 64,
        selected_identity_sha256="3" * 64, environment_steps=17)


def _candidate_lock(**overrides) -> dict:
    bindings = {name: "a" * 64 for name in subject._REQUIRED_LOCK_BINDINGS}
    bindings.update(overrides.pop("bindings", {}))
    value = {
        "schema_version": subject.CANDIDATE_LOCK_VERSION,
        "status": "locked",
        "formal_ready": False,
        "bindings": bindings,
        "source_closure": {"selector.py": bindings["source_closure_sha256"]},
        **overrides,
    }
    # Replace the synthetic closure with a digest-consistent value.
    value["source_closure"] = {"selector.py": "b" * 64}
    value["bindings"]["source_closure_sha256"] = digest(value["source_closure"])
    value["content_sha256"] = digest(value)
    return value


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)


def test_collection_contract_and_reader_api_are_unscored():
    value = subject.contract()
    assert subject.SCHEMA_VERSION == (
        "warehouse_r41_diagnostic_rcpd_v9_outer_collection_v1")
    assert subject.STATUS == "collected_unscored"
    assert value["candidate_lock_required_before_actor_output_collection"] is True
    assert value["ordered_projection_required_row_for_row"] is True
    assert value["score_computed"] is False
    assert value["program_inference_used"] is False
    assert value["protected_final_access"] is False
    assert tuple(inspect.signature(subject.authenticate_saved_collection).parameters) == (
        "report_path", "rows_path", "projection_copy_path",
        "expected_report_sha256", "expected_rows_sha256",
        "expected_projection_copy_sha256")


def test_candidate_lock_is_validated_before_any_outer_replay(tmp_path, monkeypatch):
    lock = _candidate_lock(status="not_locked")
    path = tmp_path / "candidate_lock.json"
    path.write_text(canonical(lock) + "\n", encoding="utf-8")
    monkeypatch.setattr(subject, "_resolve_snapshot", lambda **kwargs: pytest.fail(
        "snapshot/replay boundary reached before lock authentication"))
    kwargs = dict(
        actor_path="actor", protocol_path="protocol", manifest_path="manifest",
        designation_path="designation", registry_path="registry",
        registry_report_path="registry-report",
        expected_registry_sha256="1" * 64,
        expected_registry_report_sha256="2" * 64,
        projection_path="projection", projection_receipt_path="projection-receipt",
        expected_projection_sha256="3" * 64,
        expected_projection_receipt_sha256="4" * 64,
        candidate_lock_path=path, expected_candidate_lock_sha256=file_hash(path),
        failure_closeout_path="closeout", development_rows_path="development",
        program_path="program", selector_report_path="selector", output=tmp_path / "out")
    with pytest.raises(ValueError, match="locked v9 candidate"):
        subject.build(**kwargs)


def test_candidate_lock_requires_all_hash_bindings_and_canonical_content():
    value = _candidate_lock()
    assert set(subject._validate_candidate_lock_shape(value)) == set(
        subject._REQUIRED_LOCK_BINDINGS)
    missing = deepcopy(value)
    del missing["bindings"]["program_sha256"]
    missing["content_sha256"] = digest({
        key: child for key, child in missing.items() if key != "content_sha256"})
    with pytest.raises(ValueError, match="locked v9 candidate"):
        subject._validate_candidate_lock_shape(missing)
    drifted = deepcopy(value)
    drifted["bindings"]["program_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="locked v9 candidate"):
        subject._validate_candidate_lock_shape(drifted)


def test_ordered_projection_comparison_rejects_reordered_replay():
    arrays = _arrays()
    projection = _projection(arrays)
    subject._projection_matches_arrays(
        projection=projection, arrays=arrays, registry_file_sha256="1" * 64,
        registry_content_sha256="2" * 64,
        selected_identity_sha256="3" * 64, environment_steps=17)
    order = np.asarray([1, 0, 2])
    reordered = {name: value[order] for name, value in arrays.items()}
    with pytest.raises(ValueError, match="replay order"):
        subject._projection_matches_arrays(
            projection=projection, arrays=reordered,
            registry_file_sha256="1" * 64,
            registry_content_sha256="2" * 64,
            selected_identity_sha256="3" * 64, environment_steps=17)


def test_strict_npz_loader_accepts_exact_schema_and_rejects_object_payload(
    tmp_path,
):
    arrays = _arrays()
    valid = tmp_path / "valid.npz"
    _save_npz(valid, arrays)
    loaded = subject.load_authenticated_rows(
        valid, expected_rows_sha256=file_hash(valid))
    assert set(loaded) == subject.rows_v7._FIELDS

    hostile = {name: value.copy() for name, value in arrays.items()}
    hostile["observations"] = np.asarray([[{"execute": "never"}]], dtype=object)
    object_path = tmp_path / "object.npz"
    _save_npz(object_path, hostile)
    with pytest.raises(ValueError, match="safe strict NPZ"):
        subject.load_authenticated_rows(
            object_path, expected_rows_sha256=file_hash(object_path))


def test_strict_npz_loader_rejects_extra_and_duplicate_members(tmp_path):
    arrays = _arrays()
    extra = {**arrays, "surprise": np.asarray([1], dtype=np.int8)}
    extra_path = tmp_path / "extra.npz"
    _save_npz(extra_path, extra)
    with pytest.raises(ValueError, match="safe strict NPZ"):
        subject.load_authenticated_rows(
            extra_path, expected_rows_sha256=file_hash(extra_path))

    duplicate_path = tmp_path / "duplicate.npz"
    _save_npz(duplicate_path, arrays)
    buffer = io.BytesIO()
    np.save(buffer, arrays["observations"], allow_pickle=False)
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(duplicate_path, "a") as archive:
            archive.writestr("observations.npy", buffer.getvalue())
    with pytest.raises(ValueError, match="safe strict NPZ"):
        subject.load_authenticated_rows(
            duplicate_path, expected_rows_sha256=file_hash(duplicate_path))


def test_source_closure_binds_projection_registry_and_has_no_protected_reader():
    sources = subject.producer_sources()
    assert "backend/training/warehouse_r41_diagnostic_outer_collection_v9.py" in sources
    assert "backend/training/warehouse_r41_diagnostic_outer_hash_projection_v9.py" in sources
    assert "backend/training/warehouse_r41_diagnostic_rcpd_v9_outer_split.py" in sources
    forbidden = ("fresh_final_holdout", "final_once", "explanation_audit",
                 "admission", "release_receipt")
    assert not [name for name in sources if any(token in name for token in forbidden)]


def test_build_has_lock_then_replay_then_ordered_compare_then_publish_order():
    source = inspect.getsource(subject.build)
    shape = source.index("_validate_candidate_lock_shape(candidate_lock)")
    replay = source.index("projection_api._replay_outer")
    compare = source.index("_projection_matches_arrays", replay)
    write = source.index("_write_npz", compare)
    rename = source.index("os.rename(temporary, destination)", write)
    assert shape < replay < compare < write < rename
    assert source.count("snapshot.verify()") >= 3
    assert source.count("producer_sources() != sources") >= 3
