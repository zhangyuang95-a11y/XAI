from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import zipfile

import numpy as np
import pytest

from backend.training import warehouse_r41_diagnostic_rcpd_v11_fit_selector as subject
from backend.training.warehouse_native_common import digest, file_hash


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
    assert contract["consumed_v10_outer_labels_or_probabilities_read"] is False
    assert contract["fresh_v11_outer_labels_or_probabilities_read"] is False
    assert "v11 registry" in contract["consumed_v9_outer_input"]
    assert contract["prior_outer_projection_bindings"] == [
        "file_sha256", "content_sha256"]


def test_v11_candidate_lock_binds_prior_validation_projection():
    sources = {"producer.py": "b" * 64}
    bindings = {name: "a" * 64 for name in subject._LOCK_BINDINGS}
    bindings["source_closure_sha256"] = digest(sources)
    lock = subject.make_candidate_lock(
        bindings=bindings, source_closure=sources,
        selected_config_sha256="c" * 64)
    assert lock["bindings"]["prior_outer_hash_projection_sha256"] == "a" * 64
    assert lock["bindings"][
        "prior_outer_hash_projection_content_sha256"] == "a" * 64
    boundary = lock["information_boundary"]
    assert boundary["prior_outer_hashes_used_for_validation_wins"] is True
    assert boundary["consumed_v10_outer_labels_or_probabilities_read"] is False


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
            prior_projection_path="missing",
            expected_prior_projection_sha256=wrong,
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


def _prior_projection() -> dict:
    hashes = sorted([_fingerprint("prior-a"), _fingerprint("prior-b")])
    value = {
        "version": subject.registry_api.VERSION
            + ".validation-wins-exclusion.v1",
        "source_v8_closeout_content_sha256": _fingerprint("v8-closeout"),
        "source_v8_projection_content_sha256": _fingerprint("v8-projection"),
        "source_v9_closeout_content_sha256": _fingerprint("v9-closeout"),
        "source_v9_projection_content_sha256": _fingerprint("v9-projection"),
        "source_v10_final_closeout_content_sha256": _fingerprint("v10-closeout"),
        "source_v10_projection_content_sha256": _fingerprint("v10-projection"),
        "outer_observation_hashes": hashes,
        "unique_outer_observation_hash_count": len(hashes),
        "outer_observation_hashes_sha256": digest(hashes),
        "component_unique_counts": {
            "consumed_v8": 1, "consumed_v9": 1, "consumed_v10": 1},
        "component_hashes_sha256": {
            "consumed_v8": _fingerprint("v8-hashes"),
            "consumed_v9": _fingerprint("v9-hashes"),
            "consumed_v10": _fingerprint("v10-hashes"),
        },
        "selector_rule": (
            "remove fit rows matching any prior outer observation hash before "
            "reading actions, probabilities, or raw observations"),
        "raw_observations_included": False,
        "actions_included": False,
        "probabilities_included": False,
        "labels_included": False,
        "selection_used_this_projection": False,
        "formal_ready": False,
    }
    value["content_sha256"] = digest(value)
    return value


def test_v11_prior_outer_projection_is_strict_and_label_free(tmp_path: Path):
    value = _prior_projection()
    path = tmp_path / "prior.json"
    path.write_bytes(subject._json_bytes(value))
    loaded = subject.read_prior_outer_hash_projection(
        path, expected_sha256=file_hash(path),
        expected_content_sha256=value["content_sha256"])
    assert loaded == value

    unsafe = dict(value)
    unsafe["labels_included"] = True
    unsafe["content_sha256"] = digest({
        key: child for key, child in unsafe.items() if key != "content_sha256"})
    unsafe_path = tmp_path / "unsafe.json"
    unsafe_path.write_bytes(subject._json_bytes(unsafe))
    with pytest.raises(ValueError, match="Prior outer"):
        subject.read_prior_outer_hash_projection(
            unsafe_path, expected_sha256=file_hash(unsafe_path),
            expected_content_sha256=unsafe["content_sha256"])


def test_v11_validation_union_includes_prior_and_fresh_hashes():
    shared = _fingerprint("shared")
    prior = sorted([_fingerprint("prior"), shared])
    fresh = sorted([_fingerprint("fresh"), shared])
    combined = subject.validation_hash_union(prior, fresh)
    assert combined == sorted(set(prior) | set(fresh))
    assert len(combined) == 3


def test_v11_validation_union_removes_burned_v10_1505_unique_1902_rows():
    consumed_unique = sorted(
        _fingerprint(f"consumed-v10-{index}") for index in range(1505))
    consumed_rows = consumed_unique + consumed_unique[:397]
    retained = [_fingerprint(f"retained-{index}") for index in range(11)]
    public = {
        "observation_hashes": np.asarray(
            consumed_rows + retained, dtype="S64"),
    }
    validation_union = subject.validation_hash_union(consumed_unique, [])
    keep, audit = subject.freeze_validation_wins(public, validation_union)
    kept_hashes = set(np.char.decode(
        public["observation_hashes"][keep], "ascii").astype(str))
    assert len(consumed_unique) == 1505
    assert audit["removed_rows"] == 1902
    assert audit["retained_rows"] == len(retained)
    assert not kept_hashes.intersection(consumed_unique)
