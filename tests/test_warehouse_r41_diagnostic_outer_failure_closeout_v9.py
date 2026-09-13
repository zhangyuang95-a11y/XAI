from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from backend.training import warehouse_r41_diagnostic_outer_failure_closeout_v9 as subject
from backend.training.warehouse_native_common import canonical, digest, file_hash


def _fp(label: str) -> str:
    return sha256(label.encode("ascii")).hexdigest()


def test_source_closure_contains_new_producer_and_no_protected_reader():
    sources = subject.producer_sources()
    assert "backend/training/warehouse_r41_diagnostic_outer_failure_closeout_v9.py" in sources
    assert not [path for path in sources
                if "fresh_final_holdout" in path or "final_once" in path]


def _json(path: Path, value: dict) -> None:
    path.write_text(canonical(value) + "\n", encoding="utf-8")


def _npy(value: np.ndarray) -> bytes:
    stream = BytesIO()
    np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


def _hostile_rows(path: Path) -> None:
    count = subject.OUTER_SCENE_COUNT + 1
    hashes = np.asarray([_fp(f"obs-{index}") for index in range(count)], dtype="S64")
    split = np.asarray([False] + [True] * subject.OUTER_SCENE_COUNT, dtype=np.bool_)
    scenes = np.asarray(
        [_fp("fit-scene")] + [_fp(f"outer-scene-{index}")
                              for index in range(subject.OUTER_SCENE_COUNT)],
        dtype="S64",
    )
    safe = {
        "observation_hashes.npy": _npy(hashes),
        "split_validation.npy": _npy(split),
        "scene_fingerprints.npy": _npy(scenes),
        "kinds.npy": _npy(np.asarray(["ordinary"] * count, dtype="S16")),
        "anchor_ids.npy": _npy(np.asarray([""] * count, dtype="S1")),
        "branch_actions.npy": _npy(np.asarray([""] * count, dtype="S1")),
        "physical_hashes.npy": _npy(
            np.asarray([_fp(f"physical-{index}") for index in range(count)],
                       dtype="S64")),
        "action_indices.npy": _npy(np.zeros(count, dtype=np.uint8)),
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name in sorted({field + ".npy" for field in subject.v8._ROW_FIELDS}):
            archive.writestr(name, safe.get(name, b"HOSTILE-PRIVATE-PAYLOAD"))


def _fixture(tmp_path: Path, monkeypatch):
    evidence = tmp_path / "candidate"
    evidence.mkdir()
    identities = []
    family_counts = (11, 11, 11, 11, 10, 10)
    index = 0
    for family_index, count in enumerate(family_counts, start=1):
        for _ in range(count):
            identities.append({
                "batch_index": index % 12,
                "family_id": f"conflict_family_{family_index:02d}",
                "seed": 100_000 + index,
                "fingerprint": _fp(f"outer-scene-{index}"),
            })
            index += 1
    registry = {
        "version": "warehouse-r41-diagnostic-rcpd-v8-fresh-outer-registry.v1",
        "status": "frozen_identity_only_pending_outer_collection",
        "selected_outer_identities": deepcopy(identities),
        "development_validation": deepcopy(identities),
        "program_access": False,
        "program_predictions_access": False,
        "final_audit_rows_access": False,
        "final_labels_used_for_selection": False,
        "formal_ready": False,
    }
    registry["content_sha256"] = digest(registry)
    _json(evidence / "development_expansion.json", registry)
    _hostile_rows(evidence / "rows.npz")
    selector_hashes = np.asarray(
        [_fp(f"obs-{index}") for index in range(1, subject.OUTER_SCENE_COUNT + 1)],
        dtype="S64")
    with zipfile.ZipFile(evidence / "fit_selector_fit_only_rows.npz", "w") as archive:
        for name in sorted({field + ".npy" for field in subject.v8._ROW_FIELDS}):
            archive.writestr(
                name, _npy(selector_hashes)
                if name == "observation_hashes.npy" else b"HOSTILE-PRIVATE-PAYLOAD")
    for name in subject.EXPECTED_V8_ARTIFACTS - {
            "development_expansion.json", "rows.npz",
            "fit_selector_fit_only_rows.npz"}:
        (evidence / name).write_bytes((name + "\n").encode("ascii"))
    artifacts = {
        name: file_hash(evidence / name)
        for name in sorted(subject.EXPECTED_V8_ARTIFACTS)
    }
    report = {
        "version": subject.v8.VERSION,
        "status": subject.v8.STATUS_FAILED,
        "explanation_eligible": False,
        "formal_ready": False,
        "bindings": {
            "actor_file_sha256": "a" * 64,
            "manifest_file_sha256": "b" * 64,
        },
        "candidate": {
            "selection_status": "failed_development_candidate",
            "gate": {
                "passed": False,
                "checks": {"overall": True, "direction_overall": False},
            },
            "metrics": {"overall": {"fidelity": 0.91}},
        },
        "execution": {
            "final_rows_accessed": False,
            "final_labels_accessed": False,
        },
        "row_accounting": {"retained_validation_rows": subject.OUTER_SCENE_COUNT},
        "validation_scene_count": subject.OUTER_SCENE_COUNT,
        "evidence_artifacts": artifacts,
    }
    _json(evidence / "report.json", report)
    report_sha = file_hash(evidence / "report.json")
    monkeypatch.setattr(subject, "producer_sources", lambda: {"subject.py": "c" * 64})
    monkeypatch.setattr(
        subject, "EXPECTED_SELECTOR_OUTER_OVERLAP_UNIQUE_HASHES",
        subject.OUTER_SCENE_COUNT)
    monkeypatch.setattr(
        subject, "EXPECTED_SELECTOR_OUTER_OVERLAP_ROWS",
        subject.OUTER_SCENE_COUNT)
    monkeypatch.setattr(subject, "EXPECTED_SELECTOR_OUTER_OVERLAP_EFFECTIVE_PAIRS", 0)
    monkeypatch.setattr(subject.v8, "read_saved_report",
                        lambda *args, **kwargs: deepcopy(report))
    args = {
        "candidate_output": evidence,
        "expected_report_sha256": report_sha,
        "actor_path": evidence / "candidate.json",
        "protocol_path": evidence / "inputs.json",
        "manifest_path": evidence / "fit_config.json",
        "designation_path": evidence / "weights_audit.json",
        "expansion_registry_path": evidence / "development_expansion.json",
        "expected_expansion_registry_sha256": artifacts["development_expansion.json"],
        "expansion_report_path": evidence / "development_expansion_report.json",
        "expected_expansion_report_sha256": artifacts["development_expansion_report.json"],
        "expected_prior_rows_report_sha256": artifacts["prior_rows_reauthentication_report.json"],
        "previous_development_path": evidence / "source_v7_report.json",
        "expected_expansion_rows_report_sha256": artifacts["expansion_rows_reauthentication_report.json"],
        "expected_selector_report_sha256": artifacts["fit_selector_report.json"],
    }
    return evidence, report, args


def test_source_closure_contains_producer_and_no_protected_input_module():
    sources = subject.producer_sources()
    assert "backend/training/warehouse_r41_diagnostic_outer_failure_closeout_v9.py" in sources
    assert not [path for path in sources
                if "fresh_final_holdout" in path or "final_once" in path]


def test_build_authenticates_25_artifacts_and_creates_permanent_anchor(
        tmp_path, monkeypatch):
    _, report, args = _fixture(tmp_path, monkeypatch)
    permanent = tmp_path / "permanent"
    permanent.mkdir()
    output = tmp_path / "closeout"
    receipt = subject.build(
        **args, permanent_registry=permanent, output=output)

    assert receipt["status"] == subject.STATUS
    assert receipt["source"]["evidence_artifact_count"] == 25
    assert receipt["source"]["evidence_artifacts"] == report["evidence_artifacts"]
    assert receipt["failure"]["failed_gate_names"] == ["direction_overall"]
    assert receipt["failure"]["selector_outer_overlap"][
        "unique_observation_hashes"] == subject.OUTER_SCENE_COUNT
    assert receipt["failure"]["selector_outer_overlap"]["outer_rows"] == (
        subject.OUTER_SCENE_COUNT)
    assert receipt["failure"]["selector_outer_overlap"][
        "effective_intervention_pairs"] == 0
    assert receipt["disposition"]["eligible_for_development_use"] is True
    assert receipt["disposition"]["eligible_for_outer_claim"] is False
    assert receipt["information_boundary"]["protected_final_access"] is False
    anchor = permanent / receipt["campaign_key"] / subject.ANCHOR_FILENAME
    published = output / subject.ANCHOR_FILENAME
    assert anchor.read_bytes() == published.read_bytes()
    saved = subject.read_saved_closeout(
        published, expected_closeout_sha256=file_hash(published),
        permanent_registry=permanent)
    assert saved == receipt

    with pytest.raises(FileExistsError, match="already has a permanent"):
        subject.build(
            **args, permanent_registry=permanent,
            output=tmp_path / "second-output")


def test_extra_or_missing_evidence_artifact_fails_closed(tmp_path, monkeypatch):
    evidence, _, args = _fixture(tmp_path, monkeypatch)
    (evidence / "unexpected.txt").write_text("unexpected", encoding="ascii")
    with pytest.raises(ValueError, match="exactly 25 artifacts"):
        subject.create_closeout(**args)


def test_artifact_hash_is_checked_after_strict_reader(tmp_path, monkeypatch):
    evidence, _, args = _fixture(tmp_path, monkeypatch)
    (evidence / "candidate.json").write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="artifact hash differs"):
        subject.create_closeout(**args)


def test_projection_opens_no_private_row_payload(tmp_path):
    path = tmp_path / "hostile.npz"
    _hostile_rows(path)
    projection = subject._outer_observation_hash_projection(
        path, expected_sha256=file_hash(path),
        expected_validation_rows=subject.OUTER_SCENE_COUNT,
        expected_validation_scenes=subject.OUTER_SCENE_COUNT)
    assert projection["unique_outer_observation_hash_count"] == subject.OUTER_SCENE_COUNT
    assert projection["raw_observations_included"] is False
    assert projection["actions_included"] is False
    assert projection["probabilities_included"] is False
    assert "HOSTILE" not in canonical(projection)


def test_nonfailed_report_cannot_be_closed(tmp_path, monkeypatch):
    _, report, args = _fixture(tmp_path, monkeypatch)
    report["status"] = subject.v8.STATUS_PASSED
    monkeypatch.setattr(subject.v8, "read_saved_report",
                        lambda *unused_args, **unused_kwargs: deepcopy(report))
    with pytest.raises(ValueError, match="failed v8 outer"):
        subject.create_closeout(**args)


def test_anchor_tamper_is_rejected(tmp_path, monkeypatch):
    _, _, args = _fixture(tmp_path, monkeypatch)
    permanent = tmp_path / "permanent"
    permanent.mkdir()
    output = tmp_path / "closeout"
    receipt = subject.build(
        **args, permanent_registry=permanent, output=output)
    anchor = permanent / receipt["campaign_key"] / subject.ANCHOR_FILENAME
    anchor.write_bytes(b"{}\n")
    published = output / subject.ANCHOR_FILENAME
    with pytest.raises(ValueError, match="anchor differs"):
        subject.read_saved_closeout(
            published, expected_closeout_sha256=file_hash(published),
            permanent_registry=permanent)
