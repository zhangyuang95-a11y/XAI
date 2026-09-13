from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from backend.training import warehouse_r41_diagnostic_expansion_collection_v8 as subject
from backend.training.warehouse_native_common import digest, file_hash


ROOT = Path(__file__).resolve().parents[1]
FRESH_OUTER = ROOT / (
    "output/warehouse_native/"
    "r41_diagnostic_rcpd_v8_fresh_outer_registry_v2_sourceclosure_20260913"
)
FRESH_REGISTRY = FRESH_OUTER / "development_expansion.json"
FRESH_REPORT = FRESH_OUTER / "report.json"
MANIFEST = ROOT / (
    "output/warehouse_native/r41_diagnostic_conflict_scenes_v3_20260912/"
    "manifest.json"
)
DESIGNATION = ROOT / (
    "output/warehouse_native/"
    "r41_diagnostic_designation_v2_sourceclosure2_20260913/"
    "diagnostic_actor_designation.json"
)


def test_collection_contract_and_api_fix_the_only_allowed_population():
    assert tuple(inspect.signature(subject.build).parameters) == (
        "actor_path", "protocol_path", "manifest_path", "designation_path",
        "expansion_registry_path", "expansion_report_path", "output")
    value = subject.contract()
    assert value["fit_collection"] == {
        "scene_offset": 192, "scene_count": 128, "dense_critical": True}
    assert value["validation_collection"] == {
        "scene_offset": 320, "scene_count": 64,
        "dense_critical": False, "critical_anchor_period": 5}
    assert value["source_layout"] == "expansion_only"
    assert value["program_access"] is False
    assert value["prior_rows_access"] is False
    assert value["retired_holdout_access"] is False
    assert value["historical_final_access"] is False
    assert value["fresh_outer_identity_frozen_before_actor_collection"] is True
    assert value["previously_exposed_outer_reused"] is False


def test_collector_pins_and_validates_the_fresh_outer_registry():
    assert subject.expansion_api.VERSION == (
        "warehouse-r41-diagnostic-rcpd-v8-fresh-outer-registry.v1")
    assert file_hash(FRESH_REGISTRY) == subject.EXPECTED_EXPANSION_REGISTRY_SHA256
    assert file_hash(FRESH_REPORT) == subject.EXPECTED_EXPANSION_REPORT_SHA256
    registry = json.loads(FRESH_REGISTRY.read_text(encoding="utf-8"))
    report = json.loads(FRESH_REPORT.read_text(encoding="utf-8"))
    assert report["version"] == registry["version"] == subject.expansion_api.VERSION
    assert report["status"] == registry["status"] == subject.expansion_api.STATUS
    assert report["registry_file_sha256"] == file_hash(FRESH_REGISTRY)
    assert report["registry_content_sha256"] == registry["content_sha256"]

    class Actor:
        artifact_sha256 = registry["bindings"]["actor_sha256"]
        metadata = {"actor_parameters_sha256": registry["bindings"][
            "actor_parameters_sha256"]}

    prior_report = {"bindings": {"development_supplement_file_sha256":
        subject.expansion_api.EXPECTED_PREVIOUS_DEVELOPMENT_SHA256}}
    fit, outer = subject.rows_api._validate_expansion_registry(
        registry, registry_path=FRESH_REGISTRY, actor=Actor(),
        manifest_path=MANIFEST, designation_path=DESIGNATION,
        prior_report=prior_report)
    assert len(fit) == 128
    assert len(outer) == 64

    tampered = deepcopy(registry)
    tampered["information_boundary"]["outer_actor_rows_collected"] = True
    tampered["content_sha256"] = digest({
        key: value for key, value in tampered.items()
        if key != "content_sha256"})
    with pytest.raises(ValueError, match="Fresh outer registry boundary"):
        subject.rows_api._validate_expansion_registry(
            tampered, registry_path=FRESH_REGISTRY, actor=Actor(),
            manifest_path=MANIFEST, designation_path=DESIGNATION,
            prior_report=prior_report)


def test_collection_rejects_old_row_or_schedule_arguments_before_execution():
    with pytest.raises(TypeError):
        subject.build(source_rows_path=Path("old-b335.npz"))
    with pytest.raises(TypeError):
        subject.build(scene_offset=0)
    with pytest.raises(TypeError):
        subject.build(dense_critical=False)


def test_source_closure_contains_collector_and_no_final_producer():
    sources = subject.producer_sources()
    assert "backend/training/warehouse_r41_diagnostic_expansion_collection_v8.py" in sources
    assert "backend/training/warehouse_r41_diagnostic_rcpd_v8_outer_split.py" in sources
    assert "backend/training/warehouse_r41_diagnostic_rcpd_v7.py" in sources
    forbidden = (
        "fresh_final_holdout", "explanation_audit", "final_once",
        "admission", "release_receipt", "warehouse_alignment_r41_diagnostic_release",
    )
    assert not [path for path in sources
                if any(token in path for token in forbidden)]


def test_saved_reader_replays_the_fixed_offsets_and_compares_every_array(
    tmp_path, monkeypatch,
):
    sources = {"collector.py": "a" * 64}
    report = {
        "version": subject.VERSION,
        "status": subject.STATUS,
        "contract": subject.contract(),
        "bindings": {},
        "collection": {},
        "sources": sources,
        "artifacts": {},
        "final_test_accessed": False,
        "final_identity_commitment_used_for_overlap_exclusion": False,
        "final_scene_geometry_access": False,
        "final_trajectories_access": False,
        "final_actor_outputs_or_labels_access": False,
        "final_used_for_fit_selection_metrics": False,
        "participant_data_accessed": False,
        "program_accessed": False,
        "runtime_action_override": False,
        "formal_ready": False,
    }
    report["content_sha256"] = digest(report)
    report_path = tmp_path / "report.json"
    report_path.write_text(subject.canonical(report) + "\n", encoding="utf-8")
    rows_path = tmp_path / "rows.npz"
    rows_path.write_bytes(b"rows")
    paths = {"saved_report": report_path, "saved_rows": rows_path,
             "actor": tmp_path / "actor", "protocol": tmp_path / "protocol",
             "manifest": tmp_path / "manifest",
             "manifest_validation": tmp_path / "validation",
             "designation": tmp_path / "designation",
             "expansion_registry": tmp_path / "registry",
             "expansion_report": tmp_path / "registry-report"}

    class Actor:
        metadata = {"feature_names": []}

    monkeypatch.setattr(subject, "producer_sources", lambda: sources)
    monkeypatch.setattr(
        subject, "_validate_registry_and_inputs",
        lambda **kwargs: (Actor(), [{"id": "f"}], [{"id": "v"}], {}, {}, {}))
    monkeypatch.setattr(subject.rows_api, "_load_npz",
                        lambda *args: {"sentinel": np.asarray([1])})
    monkeypatch.setattr(subject.rows_api, "_validate_expansion_rows",
                        lambda *args, **kwargs: "expansion_only")
    monkeypatch.setattr(subject, "R41DiagnosticPublicRelationsV8",
                        lambda names: object())
    monkeypatch.setattr(subject.manifest_binding, "build_runtime",
                        lambda **kwargs: object())
    calls = []

    def collect(runtime, scenes, *, scene_offset, dense_critical,
                progress_label=None):
        calls.append((scene_offset, dense_critical, progress_label))
        return [scene_offset], scene_offset

    monkeypatch.setattr(subject.v7, "_collect", collect)
    monkeypatch.setattr(
        subject.v7, "_rows_to_arrays",
        lambda fit, validation: ({"sentinel": np.asarray([2])}, {}))
    with pytest.raises(ValueError, match="fixed collection replay"):
        subject._validate_saved_snapshot(
            paths=paths, designation_original=tmp_path / "designation",
            component_originals={}, sources=sources)
    assert calls == [(192, True, None), (320, False, None)]


def test_publish_guard_is_repeated_after_staging_fsync():
    source = inspect.getsource(subject.build)
    fsync = source.index("os.fsync(directory_fd)")
    second_verify = source.index("snapshot.verify()", fsync)
    source_guard = source.index("producer_sources() != sources", second_verify)
    destination_guard = source.index("destination.exists()", source_guard)
    rename = source.index("os.rename(temporary, destination)", destination_guard)
    assert fsync < second_verify < source_guard < destination_guard < rename
