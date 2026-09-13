from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import zipfile

import numpy as np
import pytest

from backend.training import warehouse_r41_diagnostic_rcpd_v11_fit_selector as subject
from backend.training.warehouse_native_common import file_hash


def _fingerprint(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def _arrays(count: int = 6) -> dict[str, np.ndarray]:
    observations = np.zeros((count, 197), dtype=np.float32)
    observations[:, 0] = np.arange(count) % 5
    actions = (np.arange(count) % 5).astype(np.uint8)
    return {
        "observations": observations,
        "probabilities": np.eye(5, dtype=np.float32)[actions],
        "action_indices": actions,
        "weights": np.ones(count, dtype=np.float32),
        "observation_hashes": np.asarray([
            subject.rows_v7.legacy._obs_hash(row) for row in observations
        ], dtype="S64"),
        "scene_fingerprints": np.asarray([
            _fingerprint(f"scene-{index}") for index in range(count)
        ], dtype="S64"),
        "episode_ids": np.asarray([
            f"episode-{index}:skilled" for index in range(count)
        ], dtype="S180"),
        "frames": np.arange(count, dtype=np.int16),
        "group_bits": np.zeros(count, dtype=np.uint8),
        "kinds": np.full(count, "ordinary", dtype="S16"),
        "anchor_ids": np.full(count, "", dtype="S240"),
        "branch_actions": np.full(count, "", dtype="S8"),
        "physical_hashes": np.full(count, "", dtype="S64"),
        "source_state_hashes": np.asarray([
            _fingerprint(f"state-{index}") for index in range(count)
        ], dtype="S64"),
        "submitted_equal": np.ones(count, dtype=np.bool_),
        "trajectory_done": np.zeros(count, dtype=np.bool_),
        "split_validation": np.asarray([
            index % 2 == 0 for index in range(count)
        ], dtype=np.bool_),
    }


def _npy(value: np.ndarray) -> bytes:
    stream = BytesIO()
    np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


def _write_rows(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(subject.rows_v7._FIELDS):
            archive.writestr(name + ".npy", _npy(arrays[name]))


def test_v11_selector_uses_replacement_registry_without_consumed_v9_targets():
    contract = subject.contract()
    assert subject.VERSION == "warehouse-r41-diagnostic-rcpd-v11-fit-selector.v1"
    assert subject.registry_api.VERSION == \
        "warehouse-r41-diagnostic-rcpd-v11-fresh-outer-registry.v1"
    assert subject.projection_api.VERSION == \
        "warehouse-r41-diagnostic-outer-hash-projection.v11"
    # The model/search family stays frozen while the evidence lineage advances.
    assert subject.GRID_VERSION == \
        "warehouse-r41-diagnostic-rcpd-v9-candidate-grid.v1"
    assert subject.FROZEN_CANDIDATE_GRID_SHA256 == \
        "900578faaf4aadc4d4b0d25494d2468f654d644a3aa3d23f9b1b155dc149680e"
    assert contract["candidate_grid_sha256"] == \
        subject.FROZEN_CANDIDATE_GRID_SHA256
    assert contract["consumed_v9_outer_labels_or_probabilities_read"] is False
    assert contract["fresh_v11_outer_labels_or_probabilities_read"] is False
    assert "v11 registry" in contract["consumed_v9_outer_input"]


def test_v11_selector_rejects_a_replacement_candidate_grid_before_file_access():
    wrong = "0" * 64
    with pytest.raises(ValueError, match="pre-v9-outer candidate grid"):
        subject._resolve_snapshot(
            actor_path="missing", protocol_path="missing",
            manifest_path="missing", designation_path="missing",
            failure_closeout_path="missing",
            expected_failure_closeout_sha256=wrong,
            permanent_closeout_registry="missing",
            registry_path="missing", registry_report_path="missing",
            expected_registry_sha256=wrong,
            expected_registry_report_sha256=wrong,
            projection_path="missing", projection_receipt_path="missing",
            expected_projection_sha256=wrong,
            expected_projection_receipt_sha256=wrong,
            development_rows_path="missing", candidate_grid_path="missing",
            expected_candidate_grid_sha256=wrong,
        )


def test_v11_validation_wins_mask_precedes_private_development_read(
    tmp_path: Path, monkeypatch,
):
    arrays = _arrays()
    path = tmp_path / "development_rows.npz"
    _write_rows(path, arrays)
    events: list[str] = []
    real_freeze = subject.freeze_validation_wins

    def freeze(*args, **kwargs):
        result = real_freeze(*args, **kwargs)
        events.append("mask_frozen")
        return result

    def full(candidate: Path):
        assert candidate == path
        assert events == ["mask_frozen"]
        events.append("private_loaded")
        return {name: value.copy() for name, value in arrays.items()}

    monkeypatch.setattr(subject, "freeze_validation_wins", freeze)
    monkeypatch.setattr(subject, "_load_full_development_rows", full)
    monkeypatch.setattr(subject.rows_api, "_validate_base_shapes", lambda *args: None)
    early: list[str] = []
    retained, audit = subject.prepare_development_rows(
        path,
        expected_sha256=file_hash(path),
        outer_observation_hashes=[arrays["observation_hashes"][1].decode()],
        actor=object(),
        early_member_audit=early,
    )

    assert early == ["observation_hashes", "scene_fingerprints", "observations"]
    assert events == ["mask_frozen", "private_loaded"]
    assert len(retained["observations"]) == len(arrays["observations"]) - 1
    assert not retained["split_validation"].any()
    assert audit["removed_rows"] == 1
    assert audit["retained_fresh_outer_observation_overlap"] == 0
    assert audit["private_development_members_read_before_mask_frozen"] is False
    assert audit["private_members_read_only_after_mask_frozen"] is True
