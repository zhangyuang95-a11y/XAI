from __future__ import annotations

from copy import deepcopy
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backend.training import warehouse_r41_diagnostic_outer_hash_projection_v10 as subject
from backend.training.warehouse_native_common import digest, file_hash


def _arrays(count: int = 3) -> dict[str, np.ndarray]:
    hashes = [digest({"observation": index}) for index in range(count)]
    arrays = {
        "observations": np.arange(count * 197, dtype=np.float32).reshape(count, 197),
        "probabilities": np.tile(
            np.asarray([[.1, .2, .3, .15, .25]], dtype=np.float32), (count, 1)),
        "action_indices": np.asarray([2] * count, dtype=np.uint8),
        "weights": np.ones(count, dtype=np.float32),
        "observation_hashes": np.asarray(hashes, dtype="S64"),
        "scene_fingerprints": np.asarray([digest({"scene": 1})] * count, dtype="S64"),
        "episode_ids": np.asarray([f"episode:{index // 2}" for index in range(count)],
                                  dtype="S180"),
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
    assert set(arrays) == subject.rows_v7._FIELDS
    return arrays


def _projection(arrays: dict[str, np.ndarray] | None = None) -> dict:
    return subject.projection_from_arrays(
        arrays or _arrays(), registry_file_sha256="1" * 64,
        registry_content_sha256="2" * 64,
        selected_identity_sha256="3" * 64, environment_steps=17)


class _Actor:
    obs_dim = 197

    def logits(self, observations: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(
            [.10, .15, .20, .40, .15], dtype=np.float32)
        return np.tile(np.log(probabilities), (len(observations), 1))


def _validation_only_rows() -> tuple[list[dict], list[dict]]:
    fingerprint = digest({"fresh-scene": 1})
    scene = {"id": "diagnostic_v10_fresh_outer_0000",
             "fingerprint": fingerprint}
    rows = []
    for index, partner in enumerate(subject.rows_v7.PARTNERS):
        rows.append({
            "observation": np.full(197, index, dtype=np.float32),
            "probabilities": np.asarray(
                [.10, .15, .20, .40, .15], dtype=np.float32),
            "action": "RIGHT", "scene": fingerprint,
            "episode": f"{scene['id']}:{fingerprint}:{partner}",
            "frame": 0, "groups": (), "kind": "ordinary", "anchor": "",
            "branch_action": "", "physical_hash": "",
            "source_state_hash": digest({"state": index}),
            "submitted_equal": True, "actor_changed_pair": False,
            "trajectory_done": True,
        })
    return rows, [scene]


def test_contract_fixes_identity_first_label_blind_schedule():
    value = subject.contract()
    assert value["scene_count"] == 64
    assert value["scene_offset"] == 512
    assert value["partners"] == ["skilled", "assertive", "noisy"]
    assert value["fixed_schedule"] is True
    assert value["raw_observations_published"] is False
    assert value["actor_actions_published"] is False
    assert value["actor_probabilities_published"] is False
    assert value["program_access"] is False
    assert value["protected_final_access"] is False


def test_projection_publishes_only_hashes_and_preserves_order():
    arrays = _arrays()
    value = _projection(arrays)
    assert subject.validate_projection(value) == value
    payload = value["projection"]
    expected = list(map(str, np.char.decode(arrays["observation_hashes"], "ascii")))
    assert payload["ordered_observation_hashes"] == expected
    assert payload["unique_observation_hashes"] == sorted(set(expected))
    assert len(payload["ordered_row_identity_hashes"]) == len(expected)
    serialized = subject._json_bytes(value)
    assert b'"observations"' not in serialized
    assert b'"probabilities"' not in serialized
    assert b'"action_indices"' not in serialized


def test_validation_only_projection_does_not_invoke_fit_weight_normalisation(
    monkeypatch: pytest.MonkeyPatch,
):
    rows, scenes = _validation_only_rows()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("fit-weight normalisation must not run")

    monkeypatch.setattr(subject.rows_v7.legacy, "_sample_weights", forbidden)
    arrays, accounting = subject._projection_rows_to_arrays(rows)
    assert np.all(arrays["split_validation"])
    assert np.array_equal(
        arrays["weights"], np.ones(len(rows), dtype=np.float32))
    assert accounting["raw_train_rows"] == 0
    assert accounting["raw_validation_rows"] == len(rows)
    subject._validate_projection_replay_arrays(
        arrays, actor=_Actor(), scenes=scenes)


def test_validation_only_projection_rejects_actor_mismatch():
    rows, scenes = _validation_only_rows()
    arrays, _accounting = subject._projection_rows_to_arrays(rows)
    arrays["action_indices"][0] = np.uint8(0)
    with pytest.raises(ValueError, match="frozen Actor"):
        subject._validate_projection_replay_arrays(
            arrays, actor=_Actor(), scenes=scenes)


def test_projection_validation_rejects_order_and_unique_set_tampering():
    value = _projection()
    swapped = deepcopy(value)
    swapped["projection"]["ordered_observation_hashes"][:2] = reversed(
        swapped["projection"]["ordered_observation_hashes"][:2])
    swapped["content_sha256"] = digest({
        key: child for key, child in swapped.items() if key != "content_sha256"})
    with pytest.raises(ValueError, match="payload"):
        subject.validate_projection(swapped)

    duplicated = deepcopy(value)
    duplicated["projection"]["unique_observation_hashes"].append(
        duplicated["projection"]["unique_observation_hashes"][0])
    duplicated["content_sha256"] = digest({
        key: child for key, child in duplicated.items() if key != "content_sha256"})
    with pytest.raises(ValueError, match="payload"):
        subject.validate_projection(duplicated)


def test_registry_validation_accepts_v10_identity_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    actor_path = tmp_path / "actor.npz"
    manifest_path = tmp_path / "manifest.json"
    designation_path = tmp_path / "designation.json"
    for path, payload in (
        (actor_path, b"actor"),
        (manifest_path, b"manifest"),
        (designation_path, b"designation"),
    ):
        path.write_bytes(payload)

    actor_parameters_sha256 = "a" * 64
    fingerprint = digest({"fresh-v10-scene": 1})
    identity = {
        "batch_index": 0,
        "family_id": "conflict_family_01",
        "seed": 48100000,
        "fingerprint": fingerprint,
    }
    scene = {
        "id": "diagnostic_v10_fresh_outer_0000",
        "seed": identity["seed"],
        "family_id": identity["family_id"],
        "fingerprint": fingerprint,
    }
    sources = {
        "backend/training/warehouse_r41_diagnostic_rcpd_v10_outer_split.py": (
            "b" * 64),
    }
    bindings = {
        "actor_sha256": file_hash(actor_path),
        "actor_parameters_sha256": actor_parameters_sha256,
        "manifest_file_sha256": file_hash(manifest_path),
        "consumed_v9_attempt_closeout_file_sha256": "c" * 64,
        "consumed_v9_attempt_closeout_content_sha256": "d" * 64,
        "consumed_v9_selected_identity_sha256": "e" * 64,
    }
    # The production contract fixes 64 identities; repeat the construction
    # with distinct seeds/fingerprints and the exact v10 scene-id prefix.
    identities = []
    scenes = []
    for index in range(subject.SCENE_COUNT):
        row = dict(identity)
        row["batch_index"] = index // 22
        row["seed"] += index
        row["fingerprint"] = digest({"fresh-v10-scene": index})
        materialized = dict(scene)
        materialized.update({
            "id": f"diagnostic_v10_fresh_outer_{index:04d}",
            "seed": row["seed"],
            "fingerprint": row["fingerprint"],
        })
        identities.append(row)
        scenes.append(materialized)
    exclusion_counts = {
        "consumed_v9_outer_identities": 64,
        "union_candidate_identities_excluded": 529,
    }
    exclusion_digests = {
        "consumed_v9_outer_identities_sha256": "e" * 64,
    }
    statistics = {
        **exclusion_counts,
        "fixed_candidate_scene_count": subject.registry_api.FIXED_CANDIDATE_SCENE_COUNT,
        "remaining_candidate_scene_count": (
            subject.registry_api.EXPECTED_REMAINING_SCENE_COUNT),
        "remaining_family_counts": subject.registry_api.EXPECTED_REMAINING_FAMILY_COUNTS,
        "selected_outer_scene_count": subject.SCENE_COUNT,
        "selected_outer_family_counts": subject.registry_api.FAMILY_QUOTAS,
        "selected_exposed_seed_overlap": 0,
        "selected_exposed_fingerprint_overlap": 0,
    }
    boundary = deepcopy(subject._RECOVERY_BOUNDARY)
    registry = {
        "version": subject.registry_api.VERSION,
        "status": subject.registry_api.STATUS,
        "bindings": bindings,
        "development_outer": scenes,
        "selected_outer_identities": identities,
        "exclusion_counts": exclusion_counts,
        "exclusion_digests": exclusion_digests,
        "statistics": statistics,
        "information_boundary": boundary,
        "program_access": False,
        "program_predictions_access": False,
        "action_labels_access": False,
        "probabilities_access": False,
        "final_audit_rows_access": False,
        "producer_sources": sources,
        "producer_sources_sha256": digest(sources),
        "formal_ready": False,
    }
    registry["content_sha256"] = digest(registry)
    registry_path = tmp_path / "development_outer.json"
    registry_path.write_bytes(subject._json_bytes(registry))
    report = {
        "version": subject.registry_api.VERSION,
        "status": subject.registry_api.STATUS,
        "registry_file_sha256": file_hash(registry_path),
        "registry_content_sha256": registry["content_sha256"],
        "bindings": bindings,
        "selection": {
            "salt": subject.registry_api.SELECTION_SALT,
            "family_quotas": subject.registry_api.FAMILY_QUOTAS,
            "selected_identity_sha256": digest(identities),
            "exclusion_digests": exclusion_digests,
        },
        "statistics": statistics,
        "information_boundary": boundary,
        "producer_sources": sources,
        "producer_sources_sha256": digest(sources),
        "formal_ready": False,
    }
    report["content_sha256"] = digest(report)
    report_path = tmp_path / "report.json"
    report_path.write_bytes(subject._json_bytes(report))

    actor = SimpleNamespace(
        artifact_sha256=file_hash(actor_path),
        metadata={"actor_parameters_sha256": actor_parameters_sha256},
    )
    monkeypatch.setattr(subject, "NumPyNativeActor", lambda _path: actor)
    monkeypatch.setattr(
        subject.designation_binding,
        "read_bound_designation_snapshot",
        lambda *_args, **_kwargs: {
            "bindings": {"actor_sha256": actor.artifact_sha256},
        },
    )
    checked_actor, checked_scenes, checked_registry, checked_report = (
        subject._validate_registry(
            paths={
                "actor": actor_path,
                "manifest": manifest_path,
                "designation": designation_path,
                "registry": registry_path,
                "registry_report": report_path,
            },
            component_originals={}, designation_original=designation_path,
        )
    )
    assert checked_actor is actor
    assert checked_scenes == scenes
    assert checked_registry == registry
    assert checked_report == report
    registry["information_boundary"]["consumed_v9_identities_excluded"] = False
    registry["content_sha256"] = digest({
        key: child for key, child in registry.items() if key != "content_sha256"
    })
    registry_path.write_bytes(subject._json_bytes(registry))
    report["registry_file_sha256"] = file_hash(registry_path)
    report["registry_content_sha256"] = registry["content_sha256"]
    report["information_boundary"] = deepcopy(registry["information_boundary"])
    report["content_sha256"] = digest({
        key: child for key, child in report.items() if key != "content_sha256"
    })
    report_path.write_bytes(subject._json_bytes(report))
    with pytest.raises(ValueError, match="registry/report binding"):
        subject._validate_registry(
            paths={
                "actor": actor_path,
                "manifest": manifest_path,
                "designation": designation_path,
                "registry": registry_path,
                "registry_report": report_path,
            },
            component_originals={}, designation_original=designation_path,
        )


def test_row_identity_hash_covers_ordered_public_replay_coordinates():
    arrays = _arrays()
    before = _projection(arrays)["projection"]["ordered_row_identity_hashes"]
    changed = {name: value.copy() for name, value in arrays.items()}
    changed["frames"][1] += 1
    after = _projection(changed)["projection"]["ordered_row_identity_hashes"]
    assert before[0] == after[0]
    assert before[1] != after[1]


def test_source_closure_has_both_v10_identity_and_replay_producers_only():
    sources = subject.producer_sources()
    assert "backend/training/warehouse_r41_diagnostic_outer_hash_projection_v10.py" in sources
    assert "backend/training/warehouse_r41_diagnostic_rcpd_v10_outer_split.py" in sources
    forbidden = ("fresh_final_holdout", "final_once", "explanation_audit",
                 "admission", "release_receipt")
    assert not [name for name in sources if any(token in name for token in forbidden)]


def test_frozen_source_receipt_survives_later_checkout_evolution():
    frozen = {"producer.py": "a" * 64, "dependency.py": "b" * 64}
    assert subject._frozen_sources_valid(frozen, digest(frozen))
    assert not subject._frozen_sources_valid(
        {"producer.py": "not-a-sha"}, digest({"producer.py": "not-a-sha"}))
    assert not subject._frozen_sources_valid(frozen, "c" * 64)


def test_build_rechecks_sources_and_inputs_at_publish_boundary():
    source = inspect.getsource(subject.build)
    assert source.count("snapshot.verify()") >= 3
    assert source.count("producer_sources() != sources") >= 3
    assert source.index("_validate_registry") < source.index("_replay_outer")
    assert source.index("_replay_outer") < source.index("os.rename(temporary, destination)")
