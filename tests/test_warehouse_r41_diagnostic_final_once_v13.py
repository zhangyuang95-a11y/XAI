from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import inspect
from pathlib import Path

import numpy as np
import pytest

from backend.training import warehouse_r41_diagnostic_final_once_v13 as subject
from backend.training import warehouse_r41_diagnostic_final_materializer_v13 as materializer_api
from backend.training.warehouse_native_common import digest, file_hash


def _fp(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _content(value: dict) -> dict:
    result = deepcopy(value)
    result["content_sha256"] = digest(result)
    return result


def _scenes() -> list[dict]:
    return [{
        "id": f"diagnostic_v13_fresh_final_{index:04d}",
        "split": "fresh_final_test", "seed": 70_000 + index,
        "family_id": f"family-{index % 6}",
        "fingerprint": _fp(f"final-scene-{index}"),
    } for index in range(subject.audit_api.FINAL_SCENE_COUNT)]


def _arrays(scenes=None):
    scenes = _scenes() if scenes is None else scenes
    hashes = [_fp(f"observation-{index}") for index in range(len(scenes))]
    return {
        "observation_hashes": np.asarray(hashes, dtype="S64"),
        "scene_fingerprints": np.asarray(
            [s["fingerprint"] for s in scenes], dtype="S64"),
    }


def _projection(scenes=None, *, changed=False):
    scenes = _scenes() if scenes is None else scenes
    hashes = [_fp(f"observation-{index}") for index in range(len(scenes))]
    if changed:
        hashes[0] = _fp("wrong")
    summaries = [{
        "local_scene_index": 0, "accepted_scene_index": index,
        "scene_index": subject.FINAL_SCENE_OFFSET + index,
        "fingerprint": scene["fingerprint"], "row_count": 1,
        "ordered_observation_hashes_sha256": digest([hashes[index]]),
        "unique_observation_count": 1,
        "unique_observation_hashes_sha256": digest([hashes[index]]),
    } for index, scene in enumerate(scenes)]
    value = {
        "version": subject.final_projection_api.VERSION,
        "contract_sha256": digest(subject.final_projection_api.contract()),
        "producer_sources_sha256": digest(
            subject.final_projection_api.producer_sources()),
        "scene_offset": subject.FINAL_SCENE_OFFSET,
        "scene_count": len(scenes),
        "partners": list(subject.final_projection_api.contract()["partners"]),
        "critical_anchor_period": 5, "dense_critical": False,
        "row_count": len(hashes),
        "ordered_observation_hashes_sha256": digest(hashes),
        "unique_observation_count": len(set(hashes)),
        "unique_observation_hashes_sha256": digest(sorted(set(hashes))),
        "scene_summaries": summaries,
        "scene_summaries_sha256": digest(summaries),
        "environment_steps": len(scenes),
        "prior_observation_overlap": 0,
        "within_final_observation_overlap": 0,
        "actions_read": False, "probabilities_read": False,
        "program_access": False, "labels_read": False,
    }
    value["content_sha256"] = digest(value)
    return value


def _material(sources, anchor, *, changed_projection=False, scenes=None):
    scenes = _scenes() if scenes is None else scenes
    return _content({
        "version": subject.MATERIAL_VERSION, "status": subject.MATERIAL_STATUS,
        "claim": {"attempt_key": anchor["attempt_key"],
                  "attempt_anchor_content_sha256": anchor["content_sha256"]},
        "scenes": scenes,
        "selection": {
            "whole_scene_selection": True, "scene_count": len(scenes),
            "program_access": False, "program_predictions_access": False,
            "actor_outputs_access": False, "action_labels_access": False,
            "salt_access_after_permanent_claim": True,
            "candidate_adaptation": False, "runtime_action_override": False,
            "exact_full_trajectory_observation_screening": True,
        },
        "observation_projection": _projection(
            scenes, changed=changed_projection),
        "producer_sources": sources,
        "producer_sources_sha256": digest(sources), "formal_ready": False,
    })


def _setup(tmp_path: Path, monkeypatch):
    paths = {}
    for name in ("lock", "actor", "protocol", "manifest", "designation",
                 "program"):
        path = tmp_path / (name + (".npz" if name == "actor" else ".json"))
        path.write_text(name + "\n")
        paths[name] = path
    bindings = {
        "actor_sha256": file_hash(paths["actor"]),
        "protocol_sha256": file_hash(paths["protocol"]),
        "runtime_manifest_sha256": file_hash(paths["manifest"]),
        "designation_sha256": file_hash(paths["designation"]),
        "program_sha256": file_hash(paths["program"]),
        "public_feature_contract_sha256": _fp("feature-contract"),
        "source_closure_sha256": _fp("candidate-closure"),
    }
    lock = _content({"lock": "synthetic"})
    outer = _content({"attempt_key": _fp("outer-attempt")})
    context = {
        "lock_path": paths["lock"], "lock": lock, "bindings": bindings,
        "paths": {"actor": paths["actor"], "protocol": paths["protocol"],
                  "runtime_manifest": paths["manifest"],
                  "designation": paths["designation"],
                  "program": paths["program"]},
        "program_payload": {"program": "synthetic"},
        "actor_feature_names": tuple(f"feature-{i}" for i in range(197)),
        "actor_feature_names_sha256": _fp("actor-features"),
        "candidate_sources": {"candidate.py": _fp("candidate")},
        "runtime_sources": {"runtime.py": _fp("runtime")},
        "outer_result": outer, "outer_result_sha256": _fp("outer-result"),
        "outer_registry": {}, "outer_registry_report": {},
        "development_hashes": {_fp("development-observation")},
        "development_scenes": {_fp("development-scene")},
        "outer_hashes": {_fp("outer-observation")},
        "outer_scenes": {_fp("outer-scene")},
        "promotion_closeout": {},
        "projection_sources": subject.final_projection_api.producer_sources(),
        "projection_contract": subject.final_projection_api.contract(),
    }
    controller_sources = {"final_once.py": _fp("controller")}
    materializer_sources = {"materializer.py": _fp("materializer")}
    monkeypatch.setattr(subject, "producer_sources", lambda: controller_sources)
    monkeypatch.setattr(subject, "_official_materializer_binding",
                        lambda: (paths["program"], materializer_sources))
    monkeypatch.setattr(subject, "_authenticate_preclaim", lambda **_: context)
    monkeypatch.setattr(subject, "_collect_final_rows",
                        lambda **_: (_arrays(), len(_scenes())))
    monkeypatch.setattr(subject.audit_api, "audit_rows", lambda **kwargs: _content({
        "status": subject.audit_api.STATUS_PASSED,
        "bindings": dict(kwargs["bindings"]), "passed": True,
    }))
    monkeypatch.setattr(subject.audit_api, "validate_report",
                        lambda report, **_: report)
    permanent = tmp_path / "permanent"
    permanent.mkdir()
    output_parent = tmp_path / "output"
    output_parent.mkdir()
    common = {
        "candidate_lock_path": paths["lock"],
        "expected_candidate_lock_sha256": file_hash(paths["lock"]),
        "actor_path": paths["actor"], "protocol_path": paths["protocol"],
        "runtime_manifest_path": paths["manifest"],
        "designation_path": paths["designation"],
        "failed_outer_closeout_path": paths["designation"],
        "promotion_closeout_path": paths["designation"],
        "expected_promotion_closeout_sha256": file_hash(paths["designation"]),
        "permanent_promotion_closeout_registry": permanent,
        "fresh_outer_registry_path": paths["designation"],
        "fresh_outer_registry_report_path": paths["designation"],
        "prior_outer_hash_projection_path": paths["designation"],
        "outer_hash_projection_path": paths["designation"],
        "outer_hash_projection_receipt_path": paths["designation"],
        "development_rows_path": paths["actor"],
        "combined_promoted_rows_path": paths["actor"],
        "program_path": paths["program"],
        "selector_report_path": paths["designation"],
        "outer_result_path": paths["designation"],
        "expected_outer_result_sha256": context["outer_result_sha256"],
        "outer_permanent_registry": permanent,
        "permanent_final_registry": permanent,
        "output": output_parent / "final",
    }
    return common, context, permanent, materializer_sources


def test_claim_precedes_materializer_and_parity_artifact_is_saved(
        tmp_path, monkeypatch):
    common, _context, permanent, sources = _setup(tmp_path, monkeypatch)

    def materializer(_path, anchor, _campaign):
        assert (permanent / anchor["attempt_key"] / subject.ANCHOR_NAME).is_file()
        return _material(sources, anchor)

    monkeypatch.setattr(subject, "_run_materializer", materializer)
    result = subject.run_final_once(**common)
    assert result["status"] == subject.STATUS_PASSED
    assert result["materializer_collector_projection_parity_passed"] is True
    assert (Path(common["output"]) / subject.PARITY_NAME).is_file()
    completion = Path(common["output"]) / subject.COMPLETION_NAME
    assert subject.read_completion(
        completion, expected_completion_sha256=file_hash(completion),
        permanent_final_registry=permanent) == result


def test_preclaim_failure_never_claims_or_invokes_materializer(tmp_path, monkeypatch):
    common, _context, permanent, _sources = _setup(tmp_path, monkeypatch)
    called = []
    monkeypatch.setattr(subject, "_authenticate_preclaim",
                        lambda **_: (_ for _ in ()).throw(ValueError("no")))
    monkeypatch.setattr(subject, "_run_materializer",
                        lambda *_: called.append(True))
    with pytest.raises(ValueError, match="no"):
        subject.run_final_once(**common)
    assert called == [] and list(permanent.iterdir()) == []


def test_projection_mismatch_burns_attempt_and_cannot_retry(tmp_path, monkeypatch):
    common, _context, permanent, sources = _setup(tmp_path, monkeypatch)
    calls = []

    def materializer(_path, anchor, _campaign):
        calls.append(True)
        return _material(sources, anchor, changed_projection=True)

    monkeypatch.setattr(subject, "_run_materializer", materializer)
    with pytest.raises(RuntimeError, match="^protected_final_phase_failed$"):
        subject.run_final_once(**common)
    campaign = next(permanent.iterdir())
    assert (campaign / subject.COMPLETION_NAME).is_file()
    with pytest.raises(FileExistsError, match="already consumed"):
        subject.run_final_once(**common)
    assert len(calls) == 1


def test_material_projection_parity_normalizes_per_scene_local_index():
    scenes = _scenes()
    arrays = _arrays(scenes)
    result = subject._validate_material_projection_parity(
        _projection(scenes), arrays, deepcopy(arrays), scenes,
        environment_steps=len(scenes), replay_environment_steps=len(scenes))
    assert result["passed"] is True
    assert result["independent_replay_equal"] is True


def test_preclaim_signature_and_source_have_no_salt_or_final_input():
    signature = inspect.signature(subject._authenticate_preclaim)
    assert not any("salt" in name or name.startswith("final_")
                   for name in signature.parameters)
    run_signature = inspect.signature(subject.run_final_once)
    assert "private_salt" not in run_signature.parameters
    source = inspect.getsource(subject.run_final_once)
    assert source.index("os.mkdir(campaign") < source.index("_run_materializer(")
    assert source.index("_write_exclusive(campaign / ANCHOR_NAME") < source.index(
        "_run_materializer(")


def test_attempt_anchor_binds_new_domain_commitment_and_projector(
        tmp_path, monkeypatch):
    common, _context, permanent, sources = _setup(tmp_path, monkeypatch)
    captured = {}

    def materializer(_path, anchor, _campaign):
        captured.update(anchor)
        return _material(sources, anchor)

    monkeypatch.setattr(subject, "_run_materializer", materializer)
    subject.run_final_once(**common)
    inputs, bindings = captured["attempt_key_inputs"], captured["bindings"]
    assert inputs["private_salt_commitment"] == subject.PRIVATE_SALT_COMMITMENT
    assert inputs["private_salt_domain_sha256"] == sha256(
        subject.PRIVATE_SALT_DOMAIN).hexdigest()
    assert bindings["final_projection_source_closure_sha256"] == digest(
        subject.final_projection_api.producer_sources())
    assert bindings["final_projection_contract_sha256"] == digest(
        subject.final_projection_api.contract())
    assert len(list(permanent.iterdir())) == 1


def test_controller_anchor_is_accepted_by_v13_materializer(
        tmp_path, monkeypatch):
    common, _context, permanent, sources = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(materializer_api, "producer_sources", lambda: sources)

    def materializer(_path, anchor, campaign):
        path, authenticated = materializer_api._authenticate_claim(
            campaign / subject.ANCHOR_NAME)
        assert path.parent == campaign
        assert authenticated == anchor
        return _material(sources, anchor)

    monkeypatch.setattr(subject, "_run_materializer", materializer)
    assert subject.run_final_once(**common)["status"] == subject.STATUS_PASSED
    assert len(list(permanent.iterdir())) == 1


def test_material_validation_rejects_nonexact_screening(tmp_path, monkeypatch):
    _common, _context, _permanent, sources = _setup(tmp_path, monkeypatch)
    anchor = _content({"attempt_key": _fp("a")})
    value = _material(sources, anchor)
    value["selection"]["exact_full_trajectory_observation_screening"] = False
    value["content_sha256"] = digest({k: v for k, v in value.items()
                                       if k != "content_sha256"})
    with pytest.raises(ValueError, match="output contract"):
        subject._validate_material(
            value, anchor=anchor, materializer_sources=sources)


def test_final_collection_uses_validation_only_encoder(monkeypatch):
    monkeypatch.setattr(subject.manifest_api, "build_runtime", lambda **_: object())
    monkeypatch.setattr(subject.rows_api, "_collect",
                        lambda *_args, **_kwargs: ([{"row": 1}], 37))
    expected = {
        "observation_hashes": np.asarray([_fp("o")], dtype="S64"),
        "scene_fingerprints": np.asarray([_fp("s")], dtype="S64"),
    }
    monkeypatch.setattr(subject.projection_api, "_projection_rows_to_arrays",
                        lambda rows: (expected, {"raw_validation_rows": len(rows)}))
    arrays, steps = subject._collect_final_rows(
        actor_path=Path("actor"), protocol_path=Path("protocol"),
        manifest_path=Path("manifest"), scenes=[{"scene": 1}])
    assert arrays is expected and steps == 37


def test_official_materializer_transitive_closure_is_frozen():
    source, sources = subject._official_materializer_binding()
    assert source == subject.ROOT / subject.OFFICIAL_FINAL_MATERIALIZER_RELATIVE_PATH
    assert digest(sources) == subject.OFFICIAL_FINAL_MATERIALIZER_SOURCE_CLOSURE_SHA256
    assert sources[subject.OFFICIAL_FINAL_MATERIALIZER_RELATIVE_PATH] == file_hash(source)
