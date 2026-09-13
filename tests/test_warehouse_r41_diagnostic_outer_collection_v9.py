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
        "selection": {
            "selected_config_sha256": "c" * 64,
            "two_salt_three_fold_development_cv_passed": True,
            "fresh_outer_scored": False,
        },
        "information_boundary": {
            "candidate_locked_before_full_fresh_outer_collection": True,
            "fresh_outer_hashes_used_only_for_validation_wins": True,
            "fresh_outer_actions_or_probabilities_read": False,
            "protected_final_access": False,
            "runtime_action_override": False,
            "formal_ready": False,
        },
        **overrides,
    }
    # Replace the synthetic closure with a digest-consistent value.
    value["source_closure"] = {"selector.py": "b" * 64}
    value["bindings"]["source_closure_sha256"] = digest(value["source_closure"])
    value["content_sha256"] = digest(value)
    return value


def _passing_metrics() -> dict:
    stat = {"rows": 64, "scenes": 64, "fidelity": 0.95}
    direction = {"pairs": 64, "scenes": 64, "fidelity": 0.95}
    return {
        "overall": deepcopy(stat), "nonwait": deepcopy(stat),
        "critical": {
            name: deepcopy(stat) for name in subject.metrics_api.GROUPS
        },
        "effective_intervention_direction": {
            **deepcopy(direction),
            "by_group": {
                name: deepcopy(direction) for name in subject.metrics_api.GROUPS
            },
        },
        "mean_kl": 0.01,
    }


def _selector_report(lock: dict) -> dict:
    metrics = _passing_metrics()
    gate = subject.metrics_api._gate(metrics)
    config = {"version": "synthetic-config"}
    config_sha = digest(config)
    folds = [{
        "fold": index, "fit_rows": 128, "validation_rows": 64,
        "fit_scenes": 32, "validation_scenes": 16,
        "fit_scene_fingerprints_sha256": digest({"fit": index}),
        "validation_scene_fingerprints_sha256": digest({"validation": index}),
        "fit_diagnostics": {"fit_rows": 128},
        "complexity": {"total_nodes": 17, "maximum_tree_depth": 3},
        "metrics": deepcopy(metrics), "informational_gate": deepcopy(gate),
    } for index in range(subject.CV_FOLDS)]
    cell = {
        "family_id": "conflict_family_01", "exact_group_bits": 1,
        "pairs": 10, "scenes": 10, "fidelity": 0.95,
    }
    salts = [{
        "salt": salt, "fold_assignment_sha256": digest({"salt": salt}),
        "folds": deepcopy(folds), "aggregate_metrics": deepcopy(metrics),
        "aggregate_gate": deepcopy(gate),
        "family_exact_bit_direction": {
            "cells": {"conflict_family_01|exact_bits=001": deepcopy(cell)},
            "cell_count": 1, "minimum_fidelity": 0.95,
        },
    } for salt in subject.CV_SALTS]
    candidate = {
        "config": config, "config_sha256": config_sha, "salts": salts,
        "both_salts_pass_all_aggregate_gates": True,
        "robust_minimum_family_exact_bit_direction_fidelity": 0.95,
        "observed_capacity": {
            "maximum_total_nodes": 17, "maximum_tree_depth": 3,
            "fold_program_count": 6,
        },
    }
    grid_payload = {
        "version": subject.SELECTOR_GRID_VERSION,
        "configs": [config],
    }
    grid_payload["content_sha256"] = digest(grid_payload)
    lock["bindings"]["candidate_grid_sha256"] = sha256(
        (canonical(grid_payload) + "\n").encode("utf-8")).hexdigest()
    lock["selection"]["selected_config_sha256"] = config_sha
    retained_semantic_sha = digest({"retained": "development"})
    validation_wins = {
        "source_rows": 200, "retained_rows": 192, "removed_rows": 8,
        "source_unique_observations": 180,
        "retained_unique_observations": 172,
        "fresh_outer_unique_observations": 64,
        "retained_fresh_outer_observation_overlap": 0,
        "keep_mask_sha256": digest({"keep": True}),
        "retained_observation_hashes_sha256": digest({"retained": True}),
        "private_development_members_read_before_mask_frozen": False,
        "retained_rows_semantic_sha256": retained_semantic_sha,
        "all_retained_rows_marked_development": True,
        "private_members_read_only_after_mask_frozen": True,
        "source_archive_reauthenticated_after_private_read": True,
    }
    validation_wins["content_sha256"] = digest(validation_wins)
    final_binding_sha = digest({
        "selector": subject.SELECTOR_VERSION,
        "development_rows_sha256": lock["bindings"]["development_rows_sha256"],
        "retained_rows_semantic_sha256": retained_semantic_sha,
        "selected_config_sha256": config_sha,
        "fresh_outer_projection_sha256": lock["bindings"][
            "outer_hash_projection_sha256"],
    })
    value = {
        "version": subject.SELECTOR_VERSION, "status": subject.SELECTOR_STATUS,
        "contract": {
            "version": subject.SELECTOR_VERSION,
            "fresh_outer_labels_or_probabilities_read": False,
            "protected_final_access": False, "runtime_action_override": False,
            "formal_ready": False,
        },
        "bindings": {
            name: lock["bindings"][name] for name in subject._SELECTOR_BINDINGS
        },
        "development": {
            "failed_v8_outer_permanently_closed": True,
            "failed_v8_outer_reclassified_as_development": True,
            "validation_wins": validation_wins,
            "scene_families": {"scene_count": 48},
            "retained_scene_count": 48,
            "retained_row_count": 192,
        },
        "candidate_grid": {
            "config_count": 1, "config_sha256s": [config_sha],
            "candidate_reports": [candidate],
        },
        "selection": {
            "selected_config": config, "selected_config_sha256": config_sha,
            "selection_key": {
                "robust_minimum_family_exact_bit_direction_fidelity": 0.95,
                "maximum_total_nodes": 17, "maximum_tree_depth": 3,
            },
            "eligible_config_sha256s": [config_sha], "eligible_count": 1,
        },
        "final_fit": {
            "binding_sha256": final_binding_sha,
            "diagnostics": {"fit_rows": 192},
            "complexity": {"total_nodes": 17, "maximum_tree_depth": 3},
            "development_metrics": deepcopy(metrics), "development_gate": gate,
            "program_file_sha256": lock["bindings"]["program_sha256"],
            "runtime_roundtrip_actions_equal": True,
            "runtime_roundtrip_max_probability_error": 0.0,
            "runtime_roundtrip_rows": 192,
        },
        "sources": deepcopy(lock["source_closure"]),
        "information_boundary": {
            "fresh_outer_projection_authenticated_before_development_targets": True,
            "validation_wins_keep_mask_frozen_before_action_or_probability_read": True,
            "fresh_outer_actions_or_probabilities_read": False,
            "fresh_outer_full_collection_access": False,
            "protected_final_access": False, "runtime_action_override": False,
            "formal_ready": False,
        },
        "formal_ready": False,
    }
    value["content_sha256"] = digest(value)
    lock["content_sha256"] = digest({
        key: child for key, child in lock.items() if key != "content_sha256"
    })
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
    assert value[
        "selector_cv_and_full_development_refit_authenticated_before_collection"] is True
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


def test_failed_cv_lock_is_rejected_before_any_outer_replay(tmp_path, monkeypatch):
    lock = _candidate_lock()
    lock["selection"]["two_salt_three_fold_development_cv_passed"] = False
    lock["content_sha256"] = digest({
        key: child for key, child in lock.items() if key != "content_sha256"
    })
    path = tmp_path / "candidate_lock.json"
    path.write_text(canonical(lock) + "\n", encoding="utf-8")
    monkeypatch.setattr(subject, "_resolve_snapshot", lambda **kwargs: pytest.fail(
        "snapshot/replay boundary reached before CV-lock rejection"))
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


@pytest.mark.parametrize("mutation", ("missing_selection", "failed_cv", "bad_boundary"))
def test_candidate_lock_rejects_missing_or_false_selection_boundary(mutation):
    value = _candidate_lock()
    if mutation == "missing_selection":
        del value["selection"]
    elif mutation == "failed_cv":
        value["selection"]["two_salt_three_fold_development_cv_passed"] = False
    else:
        value["information_boundary"][
            "fresh_outer_actions_or_probabilities_read"] = True
    value["content_sha256"] = digest({
        key: child for key, child in value.items() if key != "content_sha256"
    })
    with pytest.raises(ValueError, match="locked v9 candidate"):
        subject._validate_candidate_lock_shape(value)


def test_selector_report_authentication_rejects_forged_cv_and_full_refit(tmp_path):
    lock = _candidate_lock()
    report = _selector_report(lock)
    path = tmp_path / "selector_report.json"
    path.write_text(canonical(report) + "\n", encoding="utf-8")
    lock["bindings"]["selector_report_sha256"] = file_hash(path)
    lock["content_sha256"] = digest({
        key: child for key, child in lock.items() if key != "content_sha256"
    })
    assert subject.authenticate_locked_candidate_selector(
        lock=lock, selector_report_path=path,
        expected_selector_report_sha256=file_hash(path),
        expected_program_sha256=lock["bindings"]["program_sha256"],
    ) == report

    for label, mutate in (
        ("aggregate", lambda value: value["candidate_grid"]["candidate_reports"]
         [0]["salts"][0]["aggregate_gate"].update(passed=False)),
        ("full-development", lambda value: value["final_fit"]
         ["development_gate"].update(passed=False)),
    ):
        forged = deepcopy(report)
        mutate(forged)
        forged["content_sha256"] = digest({
            key: child for key, child in forged.items() if key != "content_sha256"
        })
        forged_path = tmp_path / (label + ".json")
        forged_path.write_text(canonical(forged) + "\n", encoding="utf-8")
        forged_lock = deepcopy(lock)
        forged_lock["bindings"]["selector_report_sha256"] = file_hash(forged_path)
        forged_lock["content_sha256"] = digest({
            key: child for key, child in forged_lock.items()
            if key != "content_sha256"
        })
        with pytest.raises(ValueError, match=label):
            subject.authenticate_locked_candidate_selector(
                lock=forged_lock, selector_report_path=forged_path,
                expected_selector_report_sha256=file_hash(forged_path),
                expected_program_sha256=forged_lock["bindings"]["program_sha256"],
            )

    missing = deepcopy(report)
    del missing["final_fit"]
    missing["content_sha256"] = digest({
        key: child for key, child in missing.items() if key != "content_sha256"
    })
    missing_path = tmp_path / "missing.json"
    missing_path.write_text(canonical(missing) + "\n", encoding="utf-8")
    missing_lock = deepcopy(lock)
    missing_lock["bindings"]["selector_report_sha256"] = file_hash(missing_path)
    missing_lock["content_sha256"] = digest({
        key: child for key, child in missing_lock.items() if key != "content_sha256"
    })
    with pytest.raises(ValueError, match="identity or bindings"):
        subject.authenticate_locked_candidate_selector(
            lock=missing_lock, selector_report_path=missing_path,
            expected_selector_report_sha256=file_hash(missing_path),
            expected_program_sha256=missing_lock["bindings"]["program_sha256"],
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (("same_salt_assignment", "salted assignments"),
     ("missing_validation_wins", "validation-wins trace"),
     ("forged_candidate_grid", "candidate-grid binding"),
     ("different_locked_selection", "chosen candidate")),
)
def test_selector_report_rejects_forged_trace_before_collection(
        tmp_path, mutation, message):
    lock = _candidate_lock()
    report = _selector_report(lock)
    if mutation == "same_salt_assignment":
        candidates = report["candidate_grid"]["candidate_reports"]
        candidates[0]["salts"][1]["fold_assignment_sha256"] = (
            candidates[0]["salts"][0]["fold_assignment_sha256"])
    elif mutation == "missing_validation_wins":
        del report["development"]["validation_wins"][
            "retained_rows_semantic_sha256"]
        trace = report["development"]["validation_wins"]
        trace["content_sha256"] = digest({
            key: child for key, child in trace.items() if key != "content_sha256"
        })
    elif mutation == "forged_candidate_grid":
        changed = digest({"different": "grid"})
        lock["bindings"]["candidate_grid_sha256"] = changed
        report["bindings"]["candidate_grid_sha256"] = changed
    else:
        lock["selection"]["selected_config_sha256"] = digest(
            {"different": "selection"})
    report["content_sha256"] = digest({
        key: child for key, child in report.items() if key != "content_sha256"
    })
    path = tmp_path / (mutation + ".json")
    path.write_text(canonical(report) + "\n", encoding="utf-8")
    lock["bindings"]["selector_report_sha256"] = file_hash(path)
    lock["content_sha256"] = digest({
        key: child for key, child in lock.items() if key != "content_sha256"
    })
    with pytest.raises(ValueError, match=message):
        subject.authenticate_locked_candidate_selector(
            lock=lock, selector_report_path=path,
            expected_selector_report_sha256=file_hash(path),
            expected_program_sha256=lock["bindings"]["program_sha256"],
        )


def test_candidate_bindings_include_registry_report_and_projection_receipt(
        tmp_path, monkeypatch):
    names = (
        "actor", "protocol", "manifest", "designation", "failure_closeout",
        "registry", "registry_report", "projection", "projection_receipt",
        "development_rows", "program", "selector_report",
    )
    paths = {}
    for name in names:
        path = tmp_path / name
        path.write_text(name + "\n", encoding="ascii")
        paths[name] = path
    lock = _candidate_lock(bindings={
        "actor_sha256": file_hash(paths["actor"]),
        "protocol_sha256": file_hash(paths["protocol"]),
        "runtime_manifest_sha256": file_hash(paths["manifest"]),
        "designation_sha256": file_hash(paths["designation"]),
        "failed_outer_closeout_sha256": file_hash(paths["failure_closeout"]),
        "fresh_outer_registry_sha256": file_hash(paths["registry"]),
        "fresh_outer_registry_report_sha256": file_hash(paths["registry_report"]),
        "outer_hash_projection_sha256": file_hash(paths["projection"]),
        "outer_hash_projection_receipt_sha256": file_hash(
            paths["projection_receipt"]),
        "development_rows_sha256": file_hash(paths["development_rows"]),
        "program_sha256": file_hash(paths["program"]),
        "selector_report_sha256": file_hash(paths["selector_report"]),
    })
    monkeypatch.setattr(
        subject, "authenticate_locked_candidate_selector", lambda **unused: {})
    assert subject._validate_candidate_bindings(lock=lock, paths=paths) == lock[
        "bindings"]
    paths["registry_report"].write_text("changed\n", encoding="ascii")
    with pytest.raises(ValueError, match="artifact binding differs"):
        subject._validate_candidate_bindings(lock=lock, paths=paths)


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
    selector = source.index("_validate_candidate_bindings")
    replay = source.index("projection_api._replay_outer")
    compare = source.index("_projection_matches_arrays", replay)
    write = source.index("_write_npz", compare)
    rename = source.index("os.rename(temporary, destination)", write)
    assert shape < selector < replay < compare < write < rename
    assert source.count("snapshot.verify()") >= 3
    assert source.count("producer_sources() != sources") >= 3
