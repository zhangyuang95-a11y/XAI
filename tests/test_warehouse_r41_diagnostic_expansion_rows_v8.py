from __future__ import annotations

from pathlib import Path

import pytest

from backend.training import warehouse_r41_diagnostic_expansion_rows_v8 as subject
from backend.training.warehouse_native_common import file_hash


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / (
    "output/warehouse_native/"
    "r41_diagnostic_expansion_collection_v8_sourceclosure4_20260913"
)
SOURCE_COLLECTION_REPORT = SOURCE_DIR / "report.json"
SOURCE_ROWS = SOURCE_DIR / "expansion_rows.npz"
PRIOR_ROWS = ROOT / (
    "output/warehouse_native/r41_diagnostic_prior_rows_v8_sourceclosure2_20260913/"
    "rows.npz"
)


def test_contract_pins_exact_historical_rows_and_has_no_final_input():
    value = subject.contract()
    assert (file_hash(SOURCE_COLLECTION_REPORT)
            == subject.EXPECTED_SOURCE_COLLECTION_REPORT_SHA256)
    assert file_hash(SOURCE_ROWS) == subject.EXPECTED_SOURCE_ROWS_SHA256
    assert file_hash(PRIOR_ROWS) == subject.EXPECTED_PRIOR_ROWS_SHA256
    assert value["source_layout"] == "expansion_only"
    assert value["final_test_accessed"] is False
    assert value["final_identity_commitment_used_for_overlap_exclusion"] is False
    assert "irrevocable claim" in value["historical_final_overlap_check"]
    assert value["program_accessed"] is False
    assert value["runtime_action_override"] is False


def test_source_closure_has_no_release_or_final_producer():
    sources = subject.producer_sources()
    assert sources
    assert "backend/training/warehouse_r41_diagnostic_rcpd_v8_outer_split.py" in sources
    assert "scripts/build_warehouse_r41_diagnostic_designation_v2.py" in sources
    forbidden = ("admission", "release", "preflight", "final_once",
                 "fresh_final_holdout", "explanation_audit")
    assert not [path for path in sources if any(token in path for token in forbidden)]


def test_source_closure_rejects_overlapping_hash_disagreement(monkeypatch):
    monkeypatch.setattr(
        subject, "local_source_hashes", lambda paths: {"shared.py": "a" * 64})
    monkeypatch.setattr(
        subject.designation_binding.designation, "source_closure",
        lambda: {"shared.py": "b" * 64})
    with pytest.raises(RuntimeError, match="closure hash disagreement"):
        subject.producer_sources()


def test_wrong_historical_row_hash_is_rejected_before_npz_load(tmp_path, monkeypatch):
    wrong = tmp_path / "rows.npz"
    wrong.write_bytes(b"not an archive")
    with pytest.raises(ValueError, match="fresh expansion collection report"):
        subject._validate(
            actor_path=wrong, protocol_path=wrong, manifest_path=wrong,
            designation_path=wrong, expansion_registry_path=wrong,
            expected_expansion_registry_sha256="0" * 64,
            expansion_report_path=wrong,
            expected_expansion_report_sha256="0" * 64,
            previous_development_path=wrong,
            source_collection_report_path=wrong,
            source_rows_path=wrong, prior_rows_path=wrong,
        )


def test_reader_rejects_wrong_receipt_hash_before_live_reauthentication(tmp_path):
    report = tmp_path / "report.json"
    rows = tmp_path / "expansion_rows.npz"
    collection = tmp_path / "source_collection_report.json"
    report.write_text("{}\n", encoding="utf-8")
    rows.write_bytes(b"x")
    collection.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="report hash differs"):
        subject.read_saved_artifacts(
            report_path=report, rows_path=rows,
            collection_report_path=collection,
            expected_report_sha256="0" * 64,
            actor_path=rows, protocol_path=report, manifest_path=report,
            designation_path=report, expansion_registry_path=report,
            expected_expansion_registry_sha256="0" * 64,
            expansion_report_path=report,
            expected_expansion_report_sha256="0" * 64,
            previous_development_path=report,
            source_collection_report_path=collection,
            source_rows_path=rows, prior_rows_path=rows,
        )


def _build_inputs(tmp_path):
    paths = {}
    for name in (
        "actor", "protocol", "manifest", "designation",
        "expansion_registry", "expansion_report", "previous",
        "source_collection_report", "source_rows", "prior_rows",
    ):
        path = tmp_path / name
        path.write_bytes((name + "\n").encode("ascii"))
        paths[name] = path
    paths["manifest_validation"] = tmp_path / "validation.json"
    paths["manifest_validation"].write_bytes(b"validation\n")
    paths["designation_actor"] = paths["actor"]
    paths["designation_protocol"] = paths["protocol"]
    for name in ("training_ledger", "dual_evaluation", "failure_closeout"):
        paths["designation_" + name] = tmp_path / ("designation_" + name)
        paths["designation_" + name].write_bytes((name + "\n").encode("ascii"))
    return paths


def _patch_manifest_validation(paths, monkeypatch):
    monkeypatch.setattr(
        subject.manifest_binding, "EXPECTED_VALIDATION_SHA256",
        file_hash(paths["manifest_validation"]))
    components = {
        "actor": paths["actor"], "protocol": paths["protocol"],
        "training_ledger": paths["designation_training_ledger"],
        "dual_evaluation": paths["designation_dual_evaluation"],
        "failure_closeout": paths["designation_failure_closeout"],
    }
    monkeypatch.setattr(
        subject, "_resolved_designation_components", lambda path: components)

    def expected_hashes(**kwargs):
        result = {
            name: file_hash(path) for name, path in paths.items()
            if name in {
                "actor", "protocol", "manifest", "manifest_validation",
                "designation", "expansion_registry", "expansion_report",
                "previous", "source_collection_report", "source_rows",
                "prior_rows",
                "designation_training_ledger", "designation_dual_evaluation",
                "designation_failure_closeout",
            }
        }
        result["designation_actor"] = file_hash(paths["actor"])
        result["designation_protocol"] = file_hash(paths["protocol"])
        return result

    monkeypatch.setattr(subject, "_expected_snapshot_hashes", expected_hashes)


def _build_kwargs(paths, output):
    return {
        "actor_path": paths["actor"],
        "protocol_path": paths["protocol"],
        "manifest_path": paths["manifest"],
        "designation_path": paths["designation"],
        "expansion_registry_path": paths["expansion_registry"],
        "expected_expansion_registry_sha256": file_hash(
            paths["expansion_registry"]),
        "expansion_report_path": paths["expansion_report"],
        "expected_expansion_report_sha256": file_hash(paths["expansion_report"]),
        "previous_development_path": paths["previous"],
        "source_collection_report_path": paths["source_collection_report"],
        "source_rows_path": paths["source_rows"],
        "prior_rows_path": paths["prior_rows"],
        "output": output,
    }


def test_build_rejects_source_change_immediately_after_authentication(
        tmp_path, monkeypatch):
    paths = _build_inputs(tmp_path)
    output = tmp_path / "output"
    _patch_manifest_validation(paths, monkeypatch)
    calls = 0

    def changing_sources():
        nonlocal calls
        calls += 1
        return {"producer.py": ("a" if calls == 1 else "b") * 64}

    monkeypatch.setattr(subject, "producer_sources", changing_sources)
    monkeypatch.setattr(
        subject, "_validate", lambda **kwargs: ({"status": "passed"}, {}))
    with pytest.raises(RuntimeError, match="changed during semantic authentication"):
        subject.build(**_build_kwargs(paths, output))
    assert not output.exists()


def test_build_rejects_source_change_at_final_publication_check(
        tmp_path, monkeypatch):
    paths = _build_inputs(tmp_path)
    output = tmp_path / "output"
    _patch_manifest_validation(paths, monkeypatch)
    calls = 0

    def changing_sources():
        nonlocal calls
        calls += 1
        return {"producer.py": ("a" if calls <= 2 else "b") * 64}

    monkeypatch.setattr(subject, "producer_sources", changing_sources)
    monkeypatch.setattr(
        subject, "_validate", lambda **kwargs: ({"status": "passed"}, {}))
    with pytest.raises(RuntimeError, match="changed during publication"):
        subject.build(**_build_kwargs(paths, output))
    assert not output.exists()


def test_build_rejects_fixed_input_change_during_authentication(
        tmp_path, monkeypatch):
    paths = _build_inputs(tmp_path)
    output = tmp_path / "output"
    _patch_manifest_validation(paths, monkeypatch)

    def mutate_input(**kwargs):
        paths["actor"].write_bytes(b"changed\n")
        return {"status": "passed"}, {}

    monkeypatch.setattr(subject, "producer_sources", lambda: {"p.py": "a" * 64})
    monkeypatch.setattr(subject, "_validate", mutate_input)
    with pytest.raises(RuntimeError, match="fixed inputs changed"):
        subject.build(**_build_kwargs(paths, output))
    assert not output.exists()


def test_build_rejects_manifest_validation_change_at_final_publication_check(
        tmp_path, monkeypatch):
    paths = _build_inputs(tmp_path)
    output = tmp_path / "output"
    _patch_manifest_validation(paths, monkeypatch)
    original_write = subject._write_json

    def mutate_after_stage(path, value):
        original_write(path, value)
        paths["manifest_validation"].write_bytes(b"changed\n")

    monkeypatch.setattr(subject, "producer_sources", lambda: {"p.py": "a" * 64})
    monkeypatch.setattr(
        subject, "_validate", lambda **kwargs: ({"status": "passed"}, {}))
    monkeypatch.setattr(subject, "_write_json", mutate_after_stage)
    with pytest.raises(RuntimeError, match="fixed inputs changed"):
        subject.build(**_build_kwargs(paths, output))
    assert not output.exists()


@pytest.mark.parametrize(
    "component", ("training_ledger", "dual_evaluation", "failure_closeout"))
def test_build_rejects_designation_component_drift_without_output(
        component, tmp_path, monkeypatch):
    paths = _build_inputs(tmp_path)
    output = tmp_path / "output"
    _patch_manifest_validation(paths, monkeypatch)
    original_write = subject._write_json

    def mutate_after_stage(path, value):
        original_write(path, value)
        paths["designation_" + component].write_bytes(b"changed\n")

    monkeypatch.setattr(subject, "producer_sources", lambda: {"p.py": "a" * 64})
    monkeypatch.setattr(
        subject, "_validate", lambda **kwargs: ({"status": "passed"}, {}))
    monkeypatch.setattr(subject, "_write_json", mutate_after_stage)
    with pytest.raises(RuntimeError, match="fixed inputs changed"):
        subject.build(**_build_kwargs(paths, output))
    assert not output.exists()


def test_designation_build_script_source_drift_aborts_without_output(
        tmp_path, monkeypatch):
    paths = _build_inputs(tmp_path)
    output = tmp_path / "output"
    _patch_manifest_validation(paths, monkeypatch)
    calls = 0

    monkeypatch.setattr(
        subject, "local_source_hashes", lambda roots: {"producer.py": "p" * 64})

    def changing_designation_closure():
        nonlocal calls
        calls += 1
        return {
            "scripts/build_warehouse_r41_diagnostic_designation_v2.py":
                ("a" if calls == 1 else "b") * 64,
        }

    monkeypatch.setattr(
        subject.designation_binding.designation, "source_closure",
        changing_designation_closure)
    monkeypatch.setattr(
        subject, "_validate", lambda **kwargs: ({"status": "passed"}, {}))
    with pytest.raises(RuntimeError, match="changed during semantic authentication"):
        subject.build(**_build_kwargs(paths, output))
    assert not output.exists()
