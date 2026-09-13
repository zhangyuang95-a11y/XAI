from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from backend.training import warehouse_r41_diagnostic_rcpd_v11_outer_once as subject
from backend.training.warehouse_native_common import canonical, digest, file_hash


_FEATURE_NAMES = tuple(f"public.feature.{index}" for index in range(197))


def _fp(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(canonical(value) + "\n", encoding="utf-8")


def _content(value: dict) -> dict:
    value = deepcopy(value)
    value["content_sha256"] = digest(value)
    return value


def _arrays(scene_fingerprints: list[str], *, prefix: str) -> dict[str, np.ndarray]:
    count = len(scene_fingerprints)
    probabilities = np.zeros((count, 5), dtype=np.float32)
    probabilities[:, 0] = 1.0
    return {
        "observations": np.zeros((count, 197), dtype=np.float32),
        "probabilities": probabilities,
        "action_indices": np.zeros(count, dtype=np.uint8),
        "weights": np.ones(count, dtype=np.float32),
        "observation_hashes": np.asarray(
            [_fp(f"{prefix}-observation-{index}") for index in range(count)],
            dtype="S64"),
        "scene_fingerprints": np.asarray(scene_fingerprints, dtype="S64"),
        "episode_ids": np.asarray(
            [f"episode-{prefix}-{index}" for index in range(count)], dtype="S180"),
        "frames": np.zeros(count, dtype=np.int16),
        "group_bits": np.zeros(count, dtype=np.uint8),
        "kinds": np.asarray(["ordinary"] * count, dtype="S16"),
        "anchor_ids": np.asarray([""] * count, dtype="S240"),
        "branch_actions": np.asarray([""] * count, dtype="S8"),
        "physical_hashes": np.asarray([""] * count, dtype="S64"),
        "source_state_hashes": np.asarray(
            [_fp(f"{prefix}-state-{index}") for index in range(count)], dtype="S64"),
        "submitted_equal": np.ones(count, dtype=np.bool_),
        "trajectory_done": np.zeros(count, dtype=np.bool_),
        "split_validation": np.ones(count, dtype=np.bool_),
    }


def _write_rows(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with open(path, "xb") as stream:
        np.savez_compressed(stream, **arrays)


def _metrics(fidelity: float) -> dict:
    stat = {"rows": 64, "scenes": 64, "fidelity": fidelity}
    direction = {"pairs": 64, "scenes": 64, "fidelity": fidelity}
    return {
        "overall": deepcopy(stat),
        "nonwait": deepcopy(stat),
        "critical": {name: deepcopy(stat) for name in subject.metrics_api.GROUPS},
        "effective_intervention_direction": {
            **deepcopy(direction),
            "by_group": {
                name: deepcopy(direction) for name in subject.metrics_api.GROUPS
            },
        },
        "mean_kl": 0.01,
    }


def _fixture(tmp_path: Path, monkeypatch, *, fidelity: float = 0.95):
    scenes = [{
        "id": f"diagnostic_v11_fresh_outer_{index:04d}",
        "seed": 10_000 + index,
        "fingerprint": _fp(f"scene-{index}"),
        "family_id": f"conflict_family_{index % 6 + 1:02d}",
    } for index in range(subject.projection_api.SCENE_COUNT)]
    selected_identities = [{
        "batch_index": index, "family_id": row["family_id"],
        "seed": row["seed"], "fingerprint": row["fingerprint"],
    } for index, row in enumerate(scenes)]
    selected_identity_sha256 = digest(selected_identities)
    registry_sources = subject.registry_api.producer_sources()
    registry_bindings = {"actor_sha256": _fp("registry-actor")}
    registry_statistics = {"selected_outer_scene_count": len(scenes)}
    registry_boundary = {
        "identity_selection_frozen_before_materialisation": True,
        "protected_final_access": False,
    }
    exclusion_digests = {"excluded": _fp("excluded")}
    registry = _content({
        "version": subject.registry_api.VERSION,
        "status": subject.registry_api.STATUS,
        "contract": {"version": subject.registry_api.VERSION},
        "bindings": registry_bindings,
        "development_outer": scenes,
        "selected_outer_identities": selected_identities,
        "exclusion_counts": {"excluded": 1},
        "exclusion_digests": exclusion_digests,
        "statistics": registry_statistics,
        "information_boundary": registry_boundary,
        "program_access": False,
        "program_predictions_access": False,
        "action_labels_access": False,
        "probabilities_access": False,
        "final_audit_rows_access": False,
        "formal_ready": False,
        "producer_sources": registry_sources,
        "producer_sources_sha256": digest(registry_sources),
    })
    registry_path = tmp_path / "development_outer.json"
    _write_json(registry_path, registry)
    registry_report = _content({
        "version": subject.registry_api.VERSION,
        "status": subject.registry_api.STATUS,
        "registry_file_sha256": file_hash(registry_path),
        "registry_content_sha256": registry["content_sha256"],
        "bindings": registry_bindings,
        "selection": {
            "salt": subject.registry_api.SELECTION_SALT,
            "family_quotas": subject.registry_api.FAMILY_QUOTAS,
            "selected_identity_sha256": selected_identity_sha256,
            "fixed_remaining_identity_sha256": _fp("remaining-identities"),
            "exclusion_digests": exclusion_digests,
        },
        "statistics": registry_statistics,
        "information_boundary": registry_boundary,
        "producer_sources": registry_sources,
        "producer_sources_sha256": digest(registry_sources),
        "formal_ready": False,
    })
    registry_report_path = tmp_path / "registry_report.json"
    _write_json(registry_report_path, registry_report)

    outer_arrays = _arrays([row["fingerprint"] for row in scenes], prefix="outer")
    outer_rows = tmp_path / "rows.npz"
    _write_rows(outer_rows, outer_arrays)
    development_arrays = _arrays([_fp("development-scene")], prefix="development")
    development_rows = tmp_path / "development_rows.npz"
    _write_rows(development_rows, development_arrays)

    projection = subject.projection_api.projection_from_arrays(
        outer_arrays, registry_file_sha256=file_hash(registry_path),
        registry_content_sha256=registry["content_sha256"],
        selected_identity_sha256=selected_identity_sha256, environment_steps=64)
    projection_path = tmp_path / "ordered_projection.json"
    _write_json(projection_path, projection)
    projection_copy = tmp_path / "projection_copy.json"
    projection_copy.write_bytes(projection_path.read_bytes())
    projection_receipt = tmp_path / "projection_receipt.json"
    _write_json(projection_receipt, _content({"receipt": "projection"}))

    program = {
        "version": "warehouse-r41-diagnostic-public-tree-program.v9",
        "relations": {"base_feature_names": list(_FEATURE_NAMES)},
    }
    program_path = tmp_path / "program.json"
    _write_json(program_path, program)
    files = {}
    for name in ("actor", "protocol", "runtime_manifest", "designation",
                 "failed_outer_closeout", "selector_report"):
        path = tmp_path / (name + (".npz" if name == "actor" else ".json"))
        if name == "actor":
            metadata = {
                "obs_dim": 197,
                "actions": list(subject.metrics_api.ACTIONS),
                "feature_names": list(_FEATURE_NAMES),
                "action_masks": False,
                "runtime_action_override": False,
            }
            np.savez(
                path, metadata_json=np.asarray(canonical(metadata)),
                **{
                    "0.weight": np.zeros((1,), dtype=np.float32),
                    "0.bias": np.zeros((1,), dtype=np.float32),
                    "2.weight": np.zeros((1,), dtype=np.float32),
                    "2.bias": np.zeros((1,), dtype=np.float32),
                    "4.weight": np.zeros((1,), dtype=np.float32),
                    "4.bias": np.zeros((1,), dtype=np.float32),
                },
            )
        else:
            path.write_bytes((name + "\n").encode("ascii"))
        files[name] = path
    closure = subject.runtime_program_sources()
    selector = {"sources": closure}
    _write_json(files["selector_report"], selector)

    bindings = {
        "actor_sha256": file_hash(files["actor"]),
        "actor_feature_names_sha256": digest(list(_FEATURE_NAMES)),
        "public_feature_contract_sha256": digest(program["relations"]),
        "protocol_sha256": file_hash(files["protocol"]),
        "runtime_manifest_sha256": file_hash(files["runtime_manifest"]),
        "designation_sha256": file_hash(files["designation"]),
        "failed_outer_closeout_sha256": file_hash(files["failed_outer_closeout"]),
        "fresh_outer_registry_sha256": file_hash(registry_path),
        "fresh_outer_registry_report_sha256": file_hash(registry_report_path),
        "outer_hash_projection_sha256": file_hash(projection_path),
        "outer_hash_projection_receipt_sha256": file_hash(projection_receipt),
        "development_rows_sha256": file_hash(development_rows),
        "candidate_grid_sha256": _fp("candidate-grid"),
        "program_sha256": file_hash(program_path),
        "selector_report_sha256": file_hash(files["selector_report"]),
        "source_closure_sha256": digest(closure),
    }
    lock = _content({
        "schema_version": subject.LOCK_SCHEMA,
        "status": "locked", "bindings": bindings,
        "source_closure": closure, "formal_ready": False,
        "selection": {
            "selected_config_sha256": _fp("selected-config"),
            "two_salt_three_fold_development_cv_passed": True,
            "fresh_outer_scored": False,
        },
        "information_boundary": {
            "candidate_locked_before_full_fresh_outer_collection": True,
            "fresh_outer_hashes_used_only_for_validation_wins": True,
            "consumed_v9_outer_labels_or_probabilities_read": False,
            "fresh_outer_actions_or_probabilities_read": False,
            "protected_final_access": False,
            "runtime_action_override": False,
            "formal_ready": False,
        },
    })
    lock_path = tmp_path / "candidate_lock.json"
    _write_json(lock_path, lock)

    collection_report = _content({
        "version": subject.collection_api.VERSION,
        "schema_version": subject.COLLECTION_SCHEMA,
        "status": subject.COLLECTION_STATUS,
        "bindings": {
            "candidate_lock_sha256": file_hash(lock_path),
            "actor_sha256": bindings["actor_sha256"],
            "protocol_sha256": bindings["protocol_sha256"],
            "runtime_manifest_sha256": bindings["runtime_manifest_sha256"],
            "designation_sha256": bindings["designation_sha256"],
            "failed_outer_closeout_sha256": bindings["failed_outer_closeout_sha256"],
            "fresh_outer_registry_sha256": bindings["fresh_outer_registry_sha256"],
            "fresh_outer_registry_report_sha256": bindings[
                "fresh_outer_registry_report_sha256"],
            "outer_hash_projection_sha256": bindings["outer_hash_projection_sha256"],
            "outer_hash_projection_receipt_sha256": file_hash(projection_receipt),
            "development_rows_sha256": bindings["development_rows_sha256"],
            "program_sha256": bindings["program_sha256"],
            "selector_report_sha256": bindings["selector_report_sha256"],
            "rows_sha256": file_hash(outer_rows),
            "projection_copy_sha256": file_hash(projection_copy),
            "ordered_replay_sha256": projection["projection"]["ordered_replay_sha256"],
        },
        "collection": {
            "row_count": len(outer_arrays["observations"]),
            "scene_count": subject.projection_api.SCENE_COUNT,
            "scored": False,
            "all_submitted_actions_equal_policy_actions": True,
            "all_actor_probabilities_and_actions_exact": True,
        },
        "information_boundary": {
            "candidate_lock_authenticated_before_outer_replay": True,
            "selector_report_and_passed_gates_authenticated_before_outer_replay": True,
            "labels_and_probabilities_collected_only_after_candidate_lock": True,
            "outer_metrics_or_candidate_score_computed": False,
            "program_loaded_or_executed": False,
            "protected_final_access": False,
            "runtime_action_override": False,
        },
        "formal_ready": False,
    })
    collection_report_path = tmp_path / "collection_report.json"
    _write_json(collection_report_path, collection_report)

    monkeypatch.setattr(
        subject.projection_api, "read_saved_projection",
        lambda **unused: (deepcopy(projection), {
            "bindings": {
                "actor_sha256": bindings["actor_sha256"],
                "fresh_outer_registry_report_sha256": bindings[
                    "fresh_outer_registry_report_sha256"],
            },
            "content_sha256": _fp("projection-receipt-content"),
        }))
    monkeypatch.setattr(
        subject.collection_api, "authenticate_saved_collection",
        lambda **unused: deepcopy(collection_report))
    development_hashes = np.char.decode(
        development_arrays["observation_hashes"], "ascii").astype(str).tolist()
    outer_unique = sorted(set(np.char.decode(
        outer_arrays["observation_hashes"], "ascii").astype(str).tolist()))
    keep = ~np.isin(
        np.asarray(development_hashes, dtype="U64"),
        np.asarray(outer_unique, dtype="U64"))
    retained = [value for value, selected in zip(development_hashes, keep)
                if bool(selected)]
    packed = np.ascontiguousarray(keep.astype(np.uint8))
    validation_wins = {
        "source_rows": len(development_hashes),
        "retained_rows": len(retained),
        "removed_rows": int(np.sum(~keep)),
        "source_unique_observations": len(set(development_hashes)),
        "retained_unique_observations": len(set(retained)),
        "fresh_outer_unique_observations": len(outer_unique),
        "retained_fresh_outer_observation_overlap": 0,
        "keep_mask_sha256": sha256(memoryview(packed).cast("B")).hexdigest(),
        "retained_observation_hashes_sha256": digest(retained),
    }
    monkeypatch.setattr(
        subject.collection_api, "authenticate_locked_candidate_selector",
        lambda **unused: {
            "status": subject.collection_api.SELECTOR_STATUS,
            "development": {"validation_wins": deepcopy(validation_wins)},
        })
    monkeypatch.setattr(
        subject.collection_api, "load_authenticated_rows",
        lambda *unused_args, **unused_kwargs: {
            name: value.copy() for name, value in outer_arrays.items()})
    monkeypatch.setattr(
        subject.projection_api, "_validate_projection_replay_arrays",
        lambda *a, **k: None)
    monkeypatch.setattr(subject.rows_api, "_effective_pairs",
                        lambda arrays, mask: np.empty((0, 2), dtype=np.int64))
    monkeypatch.setattr(subject.metrics_api, "_pair_group_bits",
                        lambda arrays, pairs: np.empty(0, dtype=np.uint8))
    monkeypatch.setattr(subject.metrics_api, "_metrics_from_probabilities",
                        lambda *a, **k: _metrics(fidelity))

    permanent = tmp_path / "permanent"
    permanent.mkdir()
    output_parent = tmp_path / "published"
    output_parent.mkdir()
    args = {
        "candidate_lock_path": lock_path,
        "expected_candidate_lock_sha256": file_hash(lock_path),
        "actor_path": files["actor"], "protocol_path": files["protocol"],
        "runtime_manifest_path": files["runtime_manifest"],
        "designation_path": files["designation"],
        "failed_outer_closeout_path": files["failed_outer_closeout"],
        "fresh_outer_registry_path": registry_path,
        "fresh_outer_registry_report_path": registry_report_path,
        "outer_hash_projection_path": projection_path,
        "outer_hash_projection_receipt_path": projection_receipt,
        "expected_outer_hash_projection_receipt_sha256": file_hash(projection_receipt),
        "outer_hash_projection_copy_path": projection_copy,
        "expected_outer_hash_projection_copy_sha256": file_hash(projection_copy),
        "development_rows_path": development_rows,
        "program_path": program_path,
        "selector_report_path": files["selector_report"],
        "outer_collection_report_path": collection_report_path,
        "expected_outer_collection_report_sha256": file_hash(collection_report_path),
        "outer_rows_path": outer_rows,
        "expected_outer_rows_sha256": file_hash(outer_rows),
        "permanent_registry": permanent,
        "output": output_parent / "result",
    }
    return args, outer_arrays, projection, permanent


class _FakeProgram:
    action_names = tuple(subject.metrics_api.ACTIONS)
    base_feature_names = _FEATURE_NAMES

    @classmethod
    def from_dict(cls, unused):
        return cls()

    def predict_proba_batch(self, observations):
        values = np.zeros((len(observations), 5), dtype=np.float64)
        values[:, 0] = 1.0
        return values


class _FakeActor:
    def __init__(self, unused):
        self.obs_dim = 197
        self.metadata = {"feature_names": list(_FEATURE_NAMES)}


def test_source_closure_has_no_protected_final_or_holdout_module():
    sources = subject.producer_sources()
    runtime_sources = subject.runtime_program_sources()
    assert "backend/training/warehouse_r41_diagnostic_rcpd_v11_outer_once.py" in sources
    assert "backend/warehouse_r41_diagnostic_public_features_v9.py" in runtime_sources
    assert "backend/warehouse_r41_diagnostic_public_tree_program_v9.py" in runtime_sources
    assert "backend/warehouse_r41_diagnostic_boosted_tree.py" in runtime_sources
    assert not [name for name in sources
                if "fresh_final" in name or "final_once" in name]
    assert not [name for name in runtime_sources
                if "fresh_final" in name or "final_once" in name]
    assert subject.contract()["attempt_anchor_before_private_outer_read"] is True


def test_runtime_source_drift_fails_before_claim_or_private_outer_read(
        tmp_path, monkeypatch):
    args, _, _, permanent = _fixture(tmp_path, monkeypatch)
    lock_path = Path(args["candidate_lock_path"])
    selector_path = Path(args["selector_report_path"])
    lock = json.loads(lock_path.read_text("utf-8"))
    selector = json.loads(selector_path.read_text("utf-8"))
    target = "backend/warehouse_r41_diagnostic_public_features_v9.py"
    assert target in lock["source_closure"]
    changed = _fp("semantically-different-public-feature-source")
    assert changed != lock["source_closure"][target]
    lock["source_closure"][target] = changed
    selector["sources"][target] = changed
    _write_json(selector_path, selector)
    lock["bindings"]["selector_report_sha256"] = file_hash(selector_path)
    lock["bindings"]["source_closure_sha256"] = digest(lock["source_closure"])
    lock["content_sha256"] = digest({
        key: value for key, value in lock.items() if key != "content_sha256"
    })
    _write_json(lock_path, lock)
    args["expected_candidate_lock_sha256"] = file_hash(lock_path)

    monkeypatch.setattr(
        subject.collection_api, "load_authenticated_rows",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("private outer rows opened before source rejection")),
    )
    with pytest.raises(ValueError, match="runtime source closure differs"):
        subject.build(**args)
    assert list(permanent.iterdir()) == []


def test_program_actor_feature_registry_mismatch_fails_before_claim_or_private_read(
        tmp_path, monkeypatch):
    args, _, _, permanent = _fixture(tmp_path, monkeypatch)
    program_path = Path(args["program_path"])
    program = json.loads(program_path.read_text("utf-8"))
    program["relations"]["base_feature_names"][-1] = "scene_fingerprint"
    _write_json(program_path, program)
    lock_path = Path(args["candidate_lock_path"])
    lock = json.loads(lock_path.read_text("utf-8"))
    lock["bindings"]["program_sha256"] = file_hash(program_path)
    lock["bindings"]["public_feature_contract_sha256"] = digest(
        program["relations"])
    lock["content_sha256"] = digest({
        key: value for key, value in lock.items() if key != "content_sha256"
    })
    _write_json(lock_path, lock)
    args["expected_candidate_lock_sha256"] = file_hash(lock_path)

    monkeypatch.setattr(
        subject.collection_api, "load_authenticated_rows",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("private outer rows opened before registry rejection")),
    )
    with pytest.raises(ValueError, match="program and Actor feature registries differ"):
        subject.build(**args)
    assert list(permanent.iterdir()) == []


def test_selector_semantic_failure_rejects_before_claim_or_private_outer_read(
        tmp_path, monkeypatch):
    args, _, _, permanent = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        subject.collection_api, "authenticate_locked_candidate_selector",
        lambda **unused: (_ for _ in ()).throw(
            ValueError("Locked v9 selector aggregate gate differs")),
    )
    monkeypatch.setattr(
        subject.collection_api, "load_authenticated_rows",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("private outer rows opened before selector rejection")),
    )
    with pytest.raises(ValueError, match="aggregate gate"):
        subject.build(**args)
    assert list(permanent.iterdir()) == []


def test_registry_report_identity_failure_rejects_before_claim_or_private_read(
        tmp_path, monkeypatch):
    args, _, _, permanent = _fixture(tmp_path, monkeypatch)
    report_path = Path(args["fresh_outer_registry_report_path"])
    report = json.loads(report_path.read_text("utf-8"))
    report["selection"]["selected_identity_sha256"] = _fp("forged-identities")
    report["content_sha256"] = digest({
        key: child for key, child in report.items() if key != "content_sha256"
    })
    _write_json(report_path, report)
    lock_path = Path(args["candidate_lock_path"])
    lock = json.loads(lock_path.read_text("utf-8"))
    lock["bindings"]["fresh_outer_registry_report_sha256"] = file_hash(report_path)
    lock["content_sha256"] = digest({
        key: child for key, child in lock.items() if key != "content_sha256"
    })
    _write_json(lock_path, lock)
    args["expected_candidate_lock_sha256"] = file_hash(lock_path)
    monkeypatch.setattr(
        subject.collection_api, "load_authenticated_rows",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("private outer rows opened before identity rejection")),
    )
    with pytest.raises(ValueError, match="registry/report identity differs"):
        subject.build(**args)
    assert list(permanent.iterdir()) == []


def test_registry_bundle_accepts_authenticated_historical_source_closure(
        tmp_path, monkeypatch):
    args, _, _, _ = _fixture(tmp_path, monkeypatch)
    registry_path = Path(args["fresh_outer_registry_path"])
    report_path = Path(args["fresh_outer_registry_report_path"])
    bindings = {
        "fresh_outer_registry_sha256": file_hash(registry_path),
        "fresh_outer_registry_report_sha256": file_hash(report_path),
    }
    monkeypatch.setattr(
        subject.registry_api, "producer_sources",
        lambda: {"later/source.py": _fp("later-source")})
    registry, report, selected = subject._registry_bundle(
        registry_path, report_path, bindings=bindings)
    assert registry["producer_sources"] == report["producer_sources"]
    assert registry["producer_sources_sha256"] == digest(
        registry["producer_sources"])
    assert selected == report["selection"]["selected_identity_sha256"]


def test_claim_precedes_private_rows_and_program_execution_and_second_call_fails(
        tmp_path, monkeypatch):
    args, arrays, _, permanent = _fixture(tmp_path, monkeypatch)
    original_load = subject.collection_api.load_authenticated_rows

    def load_after_claim(*call_args, **call_kwargs):
        campaigns = list(permanent.iterdir())
        assert len(campaigns) == 1
        assert (campaigns[0] / subject.ANCHOR_NAME).is_file()
        return original_load(*call_args, **call_kwargs)

    class ProgramAfterClaim(_FakeProgram):
        @classmethod
        def from_dict(cls, unused):
            campaigns = list(permanent.iterdir())
            assert len(campaigns) == 1
            assert (campaigns[0] / subject.ANCHOR_NAME).is_file()
            return cls()

    monkeypatch.setattr(subject.collection_api, "load_authenticated_rows", load_after_claim)
    monkeypatch.setattr(subject, "R41DiagnosticPublicTreeProgramV9", ProgramAfterClaim)
    monkeypatch.setattr(subject, "NumPyNativeActor", _FakeActor)
    result = subject.build(**args)
    assert result["status"] == subject.STATUS_PASSED
    assert result["gate"]["passed"] is True
    assert result["row_accounting"]["runtime_action_overrides"] == 0
    assert (Path(args["output"]) / subject.RESULT_NAME).is_file()
    result_path = Path(args["output"]) / subject.RESULT_NAME
    assert subject.read_saved_result(
        result_path, expected_result_sha256=file_hash(result_path),
        permanent_registry=permanent) == result
    with pytest.raises(FileExistsError, match="already been claimed"):
        subject.build(**{**args, "output": Path(args["output"]).with_name("again")})
    assert len(arrays["observations"]) == result["row_accounting"]["rows"]


def test_failed_gate_is_permanent_and_cannot_be_retried(tmp_path, monkeypatch):
    args, _, _, permanent = _fixture(tmp_path, monkeypatch, fidelity=0.80)
    monkeypatch.setattr(subject, "R41DiagnosticPublicTreeProgramV9", _FakeProgram)
    monkeypatch.setattr(subject, "NumPyNativeActor", _FakeActor)
    result = subject.build(**args)
    assert result["status"] == subject.STATUS_FAILED
    assert result["gate"]["passed"] is False
    campaign = next(permanent.iterdir())
    stored = json.loads((campaign / subject.RESULT_NAME).read_text("utf-8"))
    assert stored["status"] == subject.STATUS_FAILED
    assert stored["execution"]["retry_permitted"] is False
    with pytest.raises(FileExistsError):
        subject.build(**{**args, "output": Path(args["output"]).with_name("again")})


def test_private_archive_failure_after_claim_is_recorded_and_burns_attempt(
        tmp_path, monkeypatch):
    args, _, _, permanent = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        subject.collection_api, "load_authenticated_rows",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("hostile private member")))
    with pytest.raises(ValueError, match="hostile private member"):
        subject.build(**args)
    campaign = next(permanent.iterdir())
    failure = json.loads((campaign / subject.RESULT_NAME).read_text("utf-8"))
    assert failure["status"] == subject.STATUS_ABORTED
    assert failure["failure"]["outer_consumed"] is True
    assert failure["failure"]["retry_permitted"] is False
    with pytest.raises(FileExistsError):
        subject.build(**{**args, "output": Path(args["output"]).with_name("again")})


def test_ordered_projection_mismatch_fails_before_claim(tmp_path, monkeypatch):
    args, _, projection, permanent = _fixture(tmp_path, monkeypatch)
    bad = deepcopy(projection)
    bad["projection"]["ordered_observation_hashes"][0] = _fp("different")
    bad["projection"]["ordered_observation_hashes_sha256"] = digest(
        bad["projection"]["ordered_observation_hashes"])
    monkeypatch.setattr(
        subject.projection_api, "read_saved_projection",
        lambda **unused: (bad, {
            "bindings": {
                "actor_sha256": file_hash(args["actor_path"]),
                "fresh_outer_registry_report_sha256": file_hash(
                    args["fresh_outer_registry_report_path"]),
            },
        }))
    with pytest.raises(ValueError, match="ordered hash projection differs"):
        subject.build(**args)
    assert list(permanent.iterdir()) == []


def test_development_observation_overlap_fails_before_claim(tmp_path, monkeypatch):
    args, outer, _, permanent = _fixture(tmp_path, monkeypatch)
    with np.load(args["development_rows_path"], allow_pickle=False) as archive:
        development = {name: archive[name].copy() for name in archive.files}
    development["observation_hashes"][0] = outer["observation_hashes"][0]
    Path(args["development_rows_path"]).unlink()
    _write_rows(Path(args["development_rows_path"]), development)
    changed = file_hash(args["development_rows_path"])
    lock = json.loads(Path(args["candidate_lock_path"]).read_text("utf-8"))
    lock["bindings"]["development_rows_sha256"] = changed
    lock["content_sha256"] = digest({key: value for key, value in lock.items()
                                      if key != "content_sha256"})
    _write_json(Path(args["candidate_lock_path"]), lock)
    args["expected_candidate_lock_sha256"] = file_hash(args["candidate_lock_path"])
    report_path = Path(args["outer_collection_report_path"])
    report = json.loads(report_path.read_text("utf-8"))
    report["bindings"]["candidate_lock_sha256"] = args[
        "expected_candidate_lock_sha256"]
    report["bindings"]["development_rows_sha256"] = changed
    report["content_sha256"] = digest({key: value for key, value in report.items()
                                        if key != "content_sha256"})
    _write_json(report_path, report)
    args["expected_outer_collection_report_sha256"] = file_hash(report_path)
    monkeypatch.setattr(
        subject.collection_api, "authenticate_saved_collection",
        lambda **unused: deepcopy(report))
    with pytest.raises(ValueError, match="validation-wins projection"):
        subject.build(**args)
    assert list(permanent.iterdir()) == []
