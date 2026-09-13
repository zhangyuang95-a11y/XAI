from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import inspect
from pathlib import Path

import numpy as np
import pytest

from backend.training import warehouse_r41_diagnostic_final_once_v9 as subject
from backend.training.warehouse_native_common import digest, file_hash


def _fp(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _content(value: dict) -> dict:
    value = deepcopy(value)
    value["content_sha256"] = digest(value)
    return value


def _scenes(prefix: str = "final") -> list[dict]:
    return [{
        "id": f"{prefix}-{index}", "seed": 70_000 + index,
        "family_id": f"family-{index % 6}",
        "fingerprint": _fp(f"{prefix}-scene-{index}"),
    } for index in range(subject.audit_api.FINAL_SCENE_COUNT)]


def _arrays() -> dict[str, np.ndarray]:
    # The orchestrator treats rows opaquely; the dedicated audit tests exercise
    # the strict v9 row schema and physical semantics.
    return {"synthetic": np.asarray([1], dtype=np.uint8)}


def _setup(tmp_path: Path, monkeypatch):
    names = ("lock", "actor", "protocol", "manifest", "designation", "program")
    paths = {}
    for name in names:
        path = tmp_path / (name + (".npz" if name == "actor" else ".json"))
        path.write_bytes((name + "\n").encode("ascii"))
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
        "paths": {
            "actor": paths["actor"], "protocol": paths["protocol"],
            "runtime_manifest": paths["manifest"], "designation": paths["designation"],
            "program": paths["program"],
        },
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
    }
    controller_sources = {"final_once.py": _fp("controller")}
    materializer_sources = {"materializer.py": _fp("materializer")}
    monkeypatch.setattr(subject, "producer_sources", lambda: controller_sources)
    monkeypatch.setattr(
        subject, "_official_materializer_binding",
        lambda: (paths["program"], materializer_sources))
    monkeypatch.setattr(subject, "_authenticate_preclaim", lambda **_kwargs: context)
    monkeypatch.setattr(subject, "_collect_final_rows",
                        lambda **_kwargs: (_arrays(), 17))

    def fake_audit(**kwargs):
        return _content({
            "status": subject.audit_api.STATUS_PASSED,
            "bindings": dict(kwargs["bindings"]), "passed": True,
        })

    monkeypatch.setattr(subject.audit_api, "audit_rows", fake_audit)
    monkeypatch.setattr(subject.audit_api, "validate_report",
                        lambda report, **_kwargs: report)
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
        "fresh_outer_registry_path": paths["designation"],
        "fresh_outer_registry_report_path": paths["designation"],
        "outer_hash_projection_path": paths["designation"],
        "outer_hash_projection_receipt_path": paths["designation"],
        "development_rows_path": paths["actor"],
        "program_path": paths["program"],
        "selector_report_path": paths["designation"],
        "outer_result_path": paths["designation"],
        "expected_outer_result_sha256": context["outer_result_sha256"],
        "outer_permanent_registry": permanent,
        "permanent_final_registry": permanent,
        "output": output_parent / "final",
    }
    return common, context, permanent, materializer_sources


def _material(materializer_sources, anchor, *, scenes=None):
    return _content({
        "version": subject.MATERIAL_VERSION, "status": subject.MATERIAL_STATUS,
        "claim": {
            "attempt_key": anchor["attempt_key"],
            "attempt_anchor_content_sha256": anchor["content_sha256"],
        },
        "scenes": _scenes() if scenes is None else scenes,
        "selection": {
            "whole_scene_selection": True,
            "scene_count": subject.audit_api.FINAL_SCENE_COUNT,
            "program_access": False, "program_predictions_access": False,
            "actor_outputs_access": False, "action_labels_access": False,
            "salt_access_after_permanent_claim": True,
            "candidate_adaptation": False, "runtime_action_override": False,
        },
        "producer_sources": materializer_sources,
        "producer_sources_sha256": digest(materializer_sources),
        "formal_ready": False,
    })


def test_claim_precedes_materialization_and_success_is_readable(tmp_path, monkeypatch):
    common, _context, permanent, materializer_sources = _setup(tmp_path, monkeypatch)

    def materializer(_source_path, anchor, _campaign):
        campaign = permanent / anchor["attempt_key"]
        assert (campaign / subject.ANCHOR_NAME).is_file()
        return _material(materializer_sources, anchor)

    monkeypatch.setattr(subject, "_run_materializer", materializer)
    result = subject.run_final_once(**common)
    assert result["status"] == subject.STATUS_PASSED
    completion_path = Path(common["output"]) / subject.COMPLETION_NAME
    assert subject.read_completion(
        completion_path, expected_completion_sha256=file_hash(completion_path),
        permanent_final_registry=permanent) == result
    assert (Path(common["output"]) / subject.ROWS_NAME).is_file()


def test_preclaim_failure_cannot_invoke_final_materializer(tmp_path, monkeypatch):
    common, _context, permanent, _sources = _setup(tmp_path, monkeypatch)
    called = False

    def materializer(_source_path, _anchor, _campaign):
        nonlocal called
        called = True
        raise AssertionError("must not run")

    monkeypatch.setattr(subject, "_run_materializer", materializer)
    monkeypatch.setattr(
        subject, "_authenticate_preclaim",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("candidate rejected")))
    with pytest.raises(ValueError, match="candidate rejected"):
        subject.run_final_once(**common)
    assert called is False
    assert list(permanent.iterdir()) == []


def test_private_failure_is_sanitized_burned_and_never_retryable(tmp_path, monkeypatch):
    common, _context, permanent, _sources = _setup(tmp_path, monkeypatch)
    calls = 0

    def materializer(_source_path, _anchor, _campaign):
        nonlocal calls
        calls += 1
        raise ValueError("SECRET-SALT-AND-FINAL-IDENTITY")

    monkeypatch.setattr(subject, "_run_materializer", materializer)
    with pytest.raises(RuntimeError, match="^protected_final_phase_failed$") as caught:
        subject.run_final_once(**common)
    assert "SECRET" not in str(caught.value)
    campaigns = list(permanent.iterdir())
    assert len(campaigns) == 1
    failed = campaigns[0] / subject.COMPLETION_NAME
    assert b"SECRET" not in failed.read_bytes()
    with pytest.raises(FileExistsError, match="already consumed"):
        subject.run_final_once(**common)
    assert calls == 1


def test_scene_overlap_burns_attempt_before_any_scoring(tmp_path, monkeypatch):
    common, context, permanent, materializer_sources = _setup(tmp_path, monkeypatch)
    called_audit = False

    def audit(**_kwargs):
        nonlocal called_audit
        called_audit = True

    monkeypatch.setattr(subject.audit_api, "audit_rows", audit)
    overlapping = _scenes()
    overlapping[0]["fingerprint"] = next(iter(context["outer_scenes"]))

    def materializer(_source_path, anchor, _campaign):
        return _material(materializer_sources, anchor, scenes=overlapping)

    monkeypatch.setattr(subject, "_run_materializer", materializer)
    with pytest.raises(RuntimeError, match="protected_final_phase_failed"):
        subject.run_final_once(**common)
    assert called_audit is False
    assert (next(permanent.iterdir()) / subject.COMPLETION_NAME).is_file()


def test_source_order_has_no_final_or_salt_preclaim_parameter():
    signature = inspect.signature(subject._authenticate_preclaim)
    assert not any("final" in name or "salt" in name for name in signature.parameters)
    source = inspect.getsource(subject.run_final_once)
    assert source.index("os.mkdir(campaign") < source.index(
        "material = dict(_run_materializer(")
    assert source.index("_write_exclusive(campaign / ANCHOR_NAME") < source.index(
        "material = dict(_run_materializer(")


def test_material_contract_rejects_program_access(tmp_path, monkeypatch):
    common, _context, permanent, materializer_sources = _setup(tmp_path, monkeypatch)

    def materializer(_source_path, anchor, _campaign):
        value = _material(materializer_sources, anchor)
        value["selection"]["program_access"] = True
        value["content_sha256"] = digest({
            key: child for key, child in value.items() if key != "content_sha256"
        })
        return value

    monkeypatch.setattr(subject, "_run_materializer", materializer)
    with pytest.raises(RuntimeError, match="protected_final_phase_failed"):
        subject.run_final_once(**common)
    assert (next(permanent.iterdir()) / subject.COMPLETION_NAME).is_file()


def test_bound_materializer_runs_only_from_existing_claim(tmp_path, monkeypatch):
    source = Path(__file__).parent / "fixtures/warehouse_r41_synthetic_final_materializer_v9.py"
    campaign = tmp_path / "claimed"
    campaign.mkdir()
    anchor = _content({
        "version": subject.VERSION + ".attempt-anchor.v1",
        "status": "final_attempt_irrevocably_claimed",
        "attempt_key": _fp("subprocess-attempt"),
    })
    (campaign / subject.ANCHOR_NAME).write_text(
        subject.canonical(anchor) + "\n", encoding="utf-8")
    monkeypatch.setenv("WAREHOUSE_R41_SYNTHETIC_FINAL_SALT", "synthetic-only")
    sources = subject._materializer_identity(source)
    value = subject._run_materializer(source, anchor, campaign)
    scenes = subject._validate_material(
        value, anchor=anchor, materializer_sources=sources)
    assert len(scenes) == subject.audit_api.FINAL_SCENE_COUNT
    assert (campaign / "materializer_output.json").is_file()


def test_arbitrary_materializer_copy_is_rejected_before_claim(
        tmp_path, monkeypatch):
    common, _context, permanent, _sources = _setup(tmp_path, monkeypatch)
    official = subject.ROOT / subject.OFFICIAL_FINAL_MATERIALIZER_RELATIVE_PATH
    copied = tmp_path / official.name
    copied.write_bytes(official.read_bytes())
    monkeypatch.setattr(
        subject, "_official_materializer_binding",
        lambda: subject._authenticate_official_materializer_source(copied))
    with pytest.raises(ValueError, match="frozen repository final materializer"):
        subject.run_final_once(**common)
    assert list(permanent.iterdir()) == []


def test_modified_official_materializer_closure_is_rejected_before_claim(
        tmp_path, monkeypatch):
    common, _context, permanent, _sources = _setup(tmp_path, monkeypatch)
    official = subject.ROOT / subject.OFFICIAL_FINAL_MATERIALIZER_RELATIVE_PATH
    live_sources = subject._materializer_identity(official)
    changed_sources = dict(live_sources)
    changed_sources[subject.OFFICIAL_FINAL_MATERIALIZER_RELATIVE_PATH] = "0" * 64
    monkeypatch.setattr(
        subject, "_materializer_identity", lambda _source: changed_sources)
    monkeypatch.setattr(
        subject, "_official_materializer_binding",
        lambda: subject._authenticate_official_materializer_source(official))
    with pytest.raises(RuntimeError, match="source closure differs"):
        subject.run_final_once(**common)
    assert list(permanent.iterdir()) == []


def test_run_final_once_has_no_caller_materializer_override():
    assert "final_materializer_source_path" not in inspect.signature(
        subject.run_final_once).parameters


def test_final_collection_uses_validation_only_row_encoder(monkeypatch):
    monkeypatch.setattr(
        subject.manifest_api, "build_runtime", lambda **unused: object())
    monkeypatch.setattr(
        subject.rows_api, "_collect",
        lambda *unused_args, **unused_kwargs: ([{"row": 1}], 37))
    monkeypatch.setattr(
        subject.rows_api, "_rows_to_arrays",
        lambda *unused_args, **unused_kwargs: pytest.fail(
            "fit-weight encoder used for validation-only final rows"))
    expected = {"split_validation": np.ones(1, dtype=np.bool_)}
    monkeypatch.setattr(
        subject.projection_api, "_projection_rows_to_arrays",
        lambda rows: (expected, {"raw_validation_rows": len(rows)}))
    arrays, steps = subject._collect_final_rows(
        actor_path=Path("actor"), protocol_path=Path("protocol"),
        manifest_path=Path("manifest"), scenes=[{"scene": 1}])
    assert arrays is expected
    assert steps == 37


def test_official_materializer_binding_matches_frozen_transitive_digest():
    source, sources = subject._official_materializer_binding()
    assert source == (
        subject.ROOT / subject.OFFICIAL_FINAL_MATERIALIZER_RELATIVE_PATH)
    assert sources[subject.OFFICIAL_FINAL_MATERIALIZER_RELATIVE_PATH] == (
        file_hash(source))
    assert digest(sources) == (
        subject.OFFICIAL_FINAL_MATERIALIZER_SOURCE_CLOSURE_SHA256)
