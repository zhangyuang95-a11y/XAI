from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import inspect
from pathlib import Path
import zipfile

import numpy as np
import pytest

from backend.training import warehouse_r41_diagnostic_rcpd_v9_fit_selector as subject
from backend.training.warehouse_native_common import canonical, digest, file_hash
from env.warehouse_native.r41_diagnostic_conflict import (
    EDGE_BY_NODES,
    NODE_BY_ID,
    conflict_family_id,
)


def _fp(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def _config(pair_pool_multiplier: float = 16.0, *, iterations: int = 2) -> dict:
    return {
        "version": subject.fit_api.CONFIG_VERSION,
        "pair_pool_multiplier": pair_pool_multiplier,
        "wait_endpoint_share": 0.62,
        "shared_pickup_replacement": {
            "version": subject.fit_api.REPLACEMENT_VERSION,
            "enabled": False,
            "group": "shared_pickup",
            "combination": "replacement",
            "route": {
                "feature_name": subject.fit_api.REPLACEMENT_ROUTE_FEATURE,
                "operator": ">", "threshold": 0.5,
            },
            "fit_source": "fit_only_public_route_rows",
            "estimator": "fit_only_weighted_majority",
            "minimum_rows_per_partition": 1,
            "minimum_scenes_per_partition": 1,
            "minimum_episodes_per_partition": 1,
        },
        "model": {
            "learning_rate": 0.1,
            "max_iter": iterations,
            "max_leaf_nodes": 4,
            "min_samples_leaf": 5,
            "l2_regularization": 0.1,
            "max_depth": 3,
            "max_bins": 32,
            "random_state": 1941,
        },
    }


def _json(path: Path, value: dict) -> None:
    path.write_text(canonical(value) + "\n", encoding="utf-8")


def _arrays(count: int = 6) -> dict[str, np.ndarray]:
    observations = np.zeros((count, 197), dtype=np.float32)
    observations[:, 0] = np.arange(count) % 5
    hashes = np.asarray(
        [subject.rows_v7.legacy._obs_hash(row) for row in observations], dtype="S64")
    scenes = np.asarray([_fp(f"scene-{index}") for index in range(count)], dtype="S64")
    actions = (np.arange(count) % 5).astype(np.uint8)
    probabilities = np.eye(5, dtype=np.float32)[actions]
    return {
        "observations": observations,
        "probabilities": probabilities,
        "action_indices": actions,
        "weights": np.ones(count, dtype=np.float32),
        "observation_hashes": hashes,
        "scene_fingerprints": scenes,
        "episode_ids": np.asarray(
            [f"episode-{index}:skilled" for index in range(count)], dtype="S180"),
        "frames": np.arange(count, dtype=np.int16),
        "group_bits": np.zeros(count, dtype=np.uint8),
        "kinds": np.full(count, "ordinary", dtype="S16"),
        "anchor_ids": np.full(count, "", dtype="S240"),
        "branch_actions": np.full(count, "", dtype="S8"),
        "physical_hashes": np.full(count, "", dtype="S64"),
        "source_state_hashes": np.asarray(
            [_fp(f"state-{index}") for index in range(count)], dtype="S64"),
        "submitted_equal": np.ones(count, dtype=np.bool_),
        "trajectory_done": np.zeros(count, dtype=np.bool_),
        "split_validation": np.asarray(
            [index % 2 == 0 for index in range(count)], dtype=np.bool_),
    }


def _npy(value: np.ndarray) -> bytes:
    stream = BytesIO()
    np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


def _write_rows(path: Path, arrays: dict[str, np.ndarray], *,
                duplicate: str | None = None) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(subject.rows_v7._FIELDS):
            archive.writestr(name + ".npy", _npy(arrays[name]))
        if duplicate is not None:
            archive.writestr(duplicate + ".npy", _npy(arrays[duplicate]))


def test_candidate_grid_is_hash_bound_normalized_and_unique(tmp_path: Path):
    value = {"version": subject.GRID_VERSION, "configs": [_config()]}
    value["content_sha256"] = digest(value)
    path = tmp_path / "grid.json"
    _json(path, value)
    read = subject.read_candidate_grid(path, expected_sha256=file_hash(path))
    assert read["configs"] == [subject.fit_api.normalize_config(_config())]

    duplicate = {"version": subject.GRID_VERSION,
                 "configs": [_config(), deepcopy(_config())]}
    duplicate["content_sha256"] = digest(duplicate)
    duplicate_path = tmp_path / "duplicate.json"
    _json(duplicate_path, duplicate)
    with pytest.raises(ValueError, match="duplicate"):
        subject.read_candidate_grid(
            duplicate_path, expected_sha256=file_hash(duplicate_path))

    replacement = _config()
    replacement["shared_pickup_replacement"]["enabled"] = True
    replacement_value = {
        "version": subject.GRID_VERSION, "configs": [replacement],
    }
    replacement_value["content_sha256"] = digest(replacement_value)
    replacement_path = tmp_path / "replacement.json"
    _json(replacement_path, replacement_value)
    restored = subject.read_candidate_grid(
        replacement_path, expected_sha256=file_hash(replacement_path))
    assert restored["configs"][0]["shared_pickup_replacement"] == \
        subject.fit_api.normalize_config(replacement)["shared_pickup_replacement"]


def test_validation_wins_is_frozen_before_any_private_member_read(
    tmp_path: Path, monkeypatch,
):
    arrays = _arrays()
    path = tmp_path / "rows.npz"
    _write_rows(path, arrays)
    events: list[str] = []
    real_freeze = subject.freeze_validation_wins

    def freeze(*args, **kwargs):
        result = real_freeze(*args, **kwargs)
        events.append("mask_frozen")
        return result

    def full(candidate):
        assert events == ["mask_frozen"]
        events.append("private_loaded")
        return {name: value.copy() for name, value in arrays.items()}

    monkeypatch.setattr(subject, "freeze_validation_wins", freeze)
    monkeypatch.setattr(subject, "_load_full_development_rows", full)
    monkeypatch.setattr(subject.rows_api, "_validate_base_shapes", lambda *args: None)
    early: list[str] = []
    retained, audit = subject.prepare_development_rows(
        path, expected_sha256=file_hash(path),
        outer_observation_hashes=[arrays["observation_hashes"][1].decode()],
        actor=object(), early_member_audit=early)
    assert early == ["observation_hashes", "scene_fingerprints", "observations"]
    assert events == ["mask_frozen", "private_loaded"]
    assert len(retained["observations"]) == len(arrays["observations"]) - 1
    assert not retained["split_validation"].any()
    assert audit["removed_rows"] == 1
    assert audit["retained_fresh_outer_observation_overlap"] == 0
    assert audit["all_retained_rows_marked_development"] is True


def test_staged_npz_reader_rejects_duplicate_and_object_members(tmp_path: Path):
    arrays = _arrays()
    duplicate = tmp_path / "duplicate.npz"
    with pytest.warns(UserWarning):
        _write_rows(duplicate, arrays, duplicate="probabilities")
    with pytest.raises(ValueError, match="contents"):
        subject.read_public_development_projection(
            duplicate, expected_sha256=file_hash(duplicate))

    hostile = {name: value.copy() for name, value in arrays.items()}
    hostile["observations"] = np.empty(
        arrays["observations"].shape, dtype=object)
    hostile["observations"].fill("private-object")
    object_path = tmp_path / "object.npz"
    with zipfile.ZipFile(object_path, "w") as archive:
        for name in sorted(subject.rows_v7._FIELDS):
            stream = BytesIO()
            np.save(stream, hostile[name], allow_pickle=True)
            archive.writestr(name + ".npy", stream.getvalue())
    with pytest.raises(ValueError, match="member|Object"):
        subject.read_public_development_projection(
            object_path, expected_sha256=file_hash(object_path))


def test_blocked_folds_keep_scenes_whole_and_balance_each_family():
    families = {
        _fp(f"{family}-{index}"): family
        for family in ("f1", "f2") for index in range(7)
    }
    first = subject.assign_blocked_scene_folds(families, salt=subject.CV_SALTS[0])
    second = subject.assign_blocked_scene_folds(families, salt=subject.CV_SALTS[1])
    assert first != second
    for assignment in (first, second):
        for family in ("f1", "f2"):
            counts = [sum(assignment[scene] == fold
                          for scene, value in families.items() if value == family)
                      for fold in range(3)]
            assert max(counts) - min(counts) <= 1


def test_scene_family_is_recomputed_from_public_task_geometry():
    nodes, edge = next(iter(EDGE_BY_NODES.items()))
    tasks = [deepcopy(NODE_BY_ID[node]) for node in nodes]
    family = conflict_family_id(tasks)

    def row(label):
        return {"fingerprint": _fp(label), "family_id": family,
                "snapshot": {"tasks": deepcopy(tasks)}}

    manifest = {
        "candidate_batches": [[row("candidate")]],
        "splits": {"train": [row("train")],
                   "conflict_validation": [row("validation")],
                   "protected_name_that_must_not_be_enumerated": object()},
    }
    required = [_fp("candidate"), _fp("train"), _fp("validation")]
    mapping, audit = subject.scene_families_from_public_geometry(
        manifest, required_fingerprints=required)
    assert mapping == {fingerprint: family for fingerprint in sorted(required)}
    assert audit["protected_split_enumerated"] is False

    bad = deepcopy(manifest)
    bad["candidate_batches"][0][0]["family_id"] = "wrong"
    with pytest.raises(ValueError, match="Stored scene family"):
        subject.scene_families_from_public_geometry(
            bad, required_fingerprints=required)


def test_candidate_calls_fit_primitive_six_times_with_fold_local_split(
    monkeypatch,
):
    count = 18
    arrays = _arrays(count)
    scenes = [value.decode() for value in arrays["scene_fingerprints"]]
    scene_families = {
        scene: f"family-{index % 6}" for index, scene in enumerate(scenes)
    }
    calls = []

    class Program:
        def complexity(self):
            return {"total_nodes": 17, "maximum_tree_depth": 3}

    def fit_program(fold_arrays, fit_mask, **kwargs):
        assert np.array_equal(fold_arrays["split_validation"], ~fit_mask)
        calls.append((set(np.asarray(scenes)[fit_mask]),
                      set(np.asarray(scenes)[~fit_mask])))
        return Program(), {"fit_rows": int(np.sum(fit_mask))}

    def predict_program(program, observations):
        actions = observations[:, 0].astype(int)
        return np.eye(5, dtype=np.float64)[actions]

    monkeypatch.setattr(
        subject, "_probability_metrics",
        lambda probabilities, arrays, mask: ({"scope": int(np.sum(mask))},
                                              {"passed": True},
                                              np.empty((0, 2), dtype=np.int64),
                                              np.empty(0, dtype=np.uint8)))
    result = subject.evaluate_candidate(
        arrays, scene_families=scene_families, relations=object(),
        config=_config(), fit_program=fit_program,
        predict_program=predict_program)
    assert len(calls) == 6
    assert all(left.isdisjoint(right) for left, right in calls)
    assert result["both_salts_pass_all_aggregate_gates"] is True
    assert result["observed_capacity"]["maximum_total_nodes"] == 17


def test_selection_uses_robust_cell_then_explicit_capacity():
    base = {
        "config": _config(), "config_sha256": _fp("a"),
        "both_salts_pass_all_aggregate_gates": True,
        "robust_minimum_family_exact_bit_direction_fidelity": 0.9,
        "observed_capacity": {"maximum_total_nodes": 30,
                              "maximum_tree_depth": 4},
    }
    lower_cell = deepcopy(base)
    lower_cell.update(config_sha256=_fp("b"),
                      robust_minimum_family_exact_bit_direction_fidelity=0.89)
    simpler = deepcopy(base)
    simpler.update(config_sha256=_fp("c"),
                   observed_capacity={"maximum_total_nodes": 20,
                                      "maximum_tree_depth": 4})
    selected = subject.select_candidate([base, lower_cell, simpler])
    assert selected["selected_config_sha256"] == _fp("c")


def test_candidate_lock_is_canonical_and_accepted_by_downstream_schema(tmp_path: Path):
    from backend.training import warehouse_r41_diagnostic_outer_collection_v9 as collection
    from backend.training import warehouse_r41_diagnostic_rcpd_v9_outer_once as outer_once

    closure = {"backend/training/selector.py": _fp("source")}
    bindings = {name: _fp(name) for name in subject._LOCK_BINDINGS}
    bindings["source_closure_sha256"] = digest(closure)
    bindings["candidate_grid_sha256"] = _fp("grid")
    lock = subject.make_candidate_lock(
        bindings=bindings, source_closure=closure,
        selected_config_sha256=_fp("config"))
    assert lock["content_sha256"] == digest({
        key: value for key, value in lock.items() if key != "content_sha256"})
    assert subject._LOCK_BINDINGS.issubset(lock["bindings"])
    assert collection._validate_candidate_lock_shape(lock) == {
        name: lock["bindings"][name] for name in collection._REQUIRED_LOCK_BINDINGS
    }
    assert outer_once.LOCK_SCHEMA == subject.LOCK_SCHEMA
    assert outer_once._LOCK_BINDINGS.issubset(lock["bindings"])
    path = tmp_path / "candidate_lock.json"
    _json(path, lock)
    _, checked, checked_bindings = outer_once._candidate_lock(
        path, expected_sha256=file_hash(path))
    assert checked == lock
    assert checked_bindings == lock["bindings"]


def test_source_closure_and_build_have_no_outer_or_final_private_reader():
    from backend.training import warehouse_r41_diagnostic_rcpd_v9_outer_once as outer_once

    sources = subject.producer_sources()
    assert "backend/training/warehouse_r41_diagnostic_rcpd_v9_fit_selector.py" in sources
    forbidden = ("outer_collection_v9", "rcpd_v9_outer_once",
                 "fresh_final_holdout", "final_once")
    assert not [name for name in sources if any(token in name for token in forbidden)]
    runtime = outer_once.runtime_program_sources()
    assert not set(runtime) - set(sources)
    assert all(sources[name] == value for name, value in runtime.items())
    source = inspect.getsource(subject.prepare_development_rows)
    assert source.index("freeze_validation_wins") < source.index(
        "_load_full_development_rows")
    build_source = inspect.getsource(subject.build)
    assert "outer_collection" not in build_source
    assert "final_holdout" not in build_source
