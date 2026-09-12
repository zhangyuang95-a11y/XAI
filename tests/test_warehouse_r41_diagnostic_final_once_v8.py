import json
from pathlib import Path
import shutil

import pytest

from backend.training import warehouse_r41_diagnostic_final_once_v8 as subject


@pytest.fixture(autouse=True)
def _provision_test_commitment(monkeypatch):
    monkeypatch.setattr(subject.holdout_api, "HOLDOUT_SALT_COMMITMENT", "9" * 64)
    monkeypatch.delenv(subject.LEDGER_ENV, raising=False)
    monkeypatch.delenv(subject.PERMANENT_ANCHOR_ENV, raising=False)
    monkeypatch.delenv(subject.SALT_FILE_ENV, raising=False)


def _ledger(tmp_path, monkeypatch):
    registry = tmp_path / "external" / "ledger"
    anchor = tmp_path / "permanent" / "campaign.anchor"
    monkeypatch.setattr(subject, "REGISTRY_ROOT", registry)
    monkeypatch.setattr(subject, "PERMANENT_ANCHOR", anchor)
    return registry, anchor


def test_contract_is_irrevocable_fixed_campaign_and_program_independent():
    contract = subject.contract()
    assert contract["attempts_per_identity"] == 1
    assert contract["attempts_per_campaign"] == 1
    assert contract["retry_after_failure"] is False
    assert contract["retry_after_interruption"] is False
    assert contract["retry_after_output_removal"] is False
    assert contract["program_fits"] == 0
    assert contract["actor_updates"] == 0
    assert "development_expansion_registry_sha256" in contract["campaign_key_inputs"]
    assert {"program", "rcpd_report", "producer_sources", "output"}.issubset(
        contract["campaign_key_excludes"])
    identity = subject._campaign_identity()
    assert identity == subject.holdout_api._campaign_identity()
    assert identity["development_expansion_registry_sha256"] == (
        subject.EXPECTED_EXPANSION_REGISTRY_SHA256)
    assert subject.campaign_key() == subject.digest(identity)


def test_placeholder_commitment_refuses_before_claim(monkeypatch):
    monkeypatch.setattr(subject.holdout_api, "HOLDOUT_SALT_COMMITMENT", "0" * 64)
    with pytest.raises(ValueError, match="not provisioned"):
        subject.campaign_key()


def test_preclaim_does_not_hash_program_report_sources_or_retired(tmp_path, monkeypatch):
    names = ("actor", "protocol", "program", "rcpd_report", "manifest",
             "designation", "selected_scenes")
    singles = {name: tmp_path / name for name in names}
    developments = [tmp_path / "supplement", tmp_path / "expansion"]
    expected = {
        singles["actor"]: subject.EXPECTED_ACTOR_SHA256,
        singles["protocol"]: subject.EXPECTED_PROTOCOL_SHA256,
        singles["manifest"]: subject.EXPECTED_MANIFEST_SHA256,
        singles["designation"]: subject.EXPECTED_DESIGNATION_SHA256,
        singles["selected_scenes"]: subject.EXPECTED_SELECTED_SCENES_SHA256,
        developments[0]: "1" * 64,
        developments[1]: subject.EXPECTED_EXPANSION_REGISTRY_SHA256,
    }
    seen = []

    def hashes(path):
        path = Path(path)
        seen.append(path)
        if path in (singles["program"], singles["rcpd_report"]):
            raise AssertionError("program/report accessed before claim")
        return expected[path]

    monkeypatch.setattr(subject, "file_hash", hashes)
    monkeypatch.setattr(
        subject, "producer_sources",
        lambda: (_ for _ in ()).throw(AssertionError("sources accessed before claim")))
    assert subject._preclaim_identity(singles, developments) == subject._campaign_identity()
    assert singles["program"] not in seen and singles["rcpd_report"] not in seen


def test_claim_writes_external_anchor_first_and_one_campaign_only(tmp_path, monkeypatch):
    registry_root, anchor = _ledger(tmp_path, monkeypatch)
    identity = subject._campaign_identity()
    key, registry, started = subject._claim(identity, output=tmp_path / "out")
    assert key == subject.campaign_key()
    assert registry == registry_root / key
    assert anchor.is_file()
    assert (registry / "attempt_started.json").is_file()
    assert started["program_evaluation_started"] is False
    assert started["permanent_anchor_sha256"] == subject.file_hash(anchor)
    with pytest.raises(ValueError, match="already_reserved_no_retry"):
        subject._claim(identity, output=tmp_path / "different-output")


def test_deleting_output_and_ledger_or_changing_sources_does_not_refund(
        tmp_path, monkeypatch):
    registry_root, anchor = _ledger(tmp_path, monkeypatch)
    identity = subject._campaign_identity()
    _, registry, _ = subject._claim(identity, output=tmp_path / "out")
    shutil.rmtree(registry_root)
    assert not registry.exists() and anchor.is_file()
    monkeypatch.setattr(subject, "producer_sources", lambda: {"changed.py": "a" * 64})
    with pytest.raises(ValueError, match="already_reserved_no_retry"):
        subject._claim(identity, output=tmp_path / "new-output")


def test_input_paths_accept_only_v1_v2_and_one_candidate_directory(tmp_path):
    candidate = tmp_path / "candidate"; candidate.mkdir()
    paths = {}
    for name in ("program.json", "report.json", "rows.npz", "prior_v7_report.json",
                 "prior_v7_rows.npz", "expansion_rows.npz", "fit_config.json"):
        path = candidate / name; path.write_bytes(name.encode()); paths[name] = path
    singles = {}
    for name in ("actor", "protocol", "manifest", "designation", "selected"):
        path = tmp_path / name; path.write_bytes(name.encode()); singles[name] = path
    dev = []
    for index in range(2):
        path = tmp_path / f"dev-{index}"; path.write_bytes(b"dev"); dev.append(path)
    retired = []
    for index in range(3):
        path = tmp_path / f"retired-{index}"; path.write_bytes(b"retired"); retired.append(path)
    kwargs = dict(
        actor_path=singles["actor"], protocol_path=singles["protocol"],
        program_path=paths["program.json"], rcpd_report_path=paths["report.json"],
        manifest_path=singles["manifest"], designation_path=singles["designation"],
        selected_scenes_path=singles["selected"], development_registry_paths=dev,
        development_rows_paths=[paths["rows.npz"]], retired_holdout_paths=retired[:2],
    )
    result = subject._input_paths(**kwargs)
    assert len(result[2]) == 1 and len(result[3]) == 2
    with pytest.raises(ValueError, match="retired v1/v2 only"):
        subject._input_paths(**{**kwargs, "retired_holdout_paths": retired})


def test_candidate_authentication_uses_same_directory_strict_refit(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate"; candidate.mkdir()
    files = {}
    for name in ("program.json", "report.json", "rows.npz", "prior_v7_report.json",
                 "prior_v7_rows.npz", "expansion_rows.npz", "fit_config.json"):
        path = candidate / name; path.write_bytes(name.encode()); files[name] = path
    external = {}
    for name in ("actor", "protocol", "manifest", "designation"):
        path = tmp_path / name; path.write_bytes(name.encode()); external[name] = path
    designation = {
        "bindings": {"actor_sha256": subject.EXPECTED_ACTOR_SHA256},
        "behavior_performance_gate_waived": True,
        "runtime_action_override": False, "formal_ready": False,
        "formal_sample_eligible": False,
    }
    external["designation"].write_text(json.dumps(designation), encoding="utf-8")
    supplement = tmp_path / "supplement.json"
    supplement.write_text(json.dumps({
        "version": subject.holdout_api.DEVELOPMENT_SUPPLEMENT_VERSION}), encoding="utf-8")
    expansion = tmp_path / "expansion.json"
    expansion.write_text(json.dumps({
        "version": subject.holdout_api.DEVELOPMENT_EXPANSION_VERSION}), encoding="utf-8")
    retired = []
    for version in subject.holdout_api.EXTERNAL_RETIRED_VERSIONS:
        path = tmp_path / (version + ".json")
        path.write_text(json.dumps({"version": version}), encoding="utf-8")
        retired.append(path)
    monkeypatch.setattr(subject, "EXPECTED_EXPANSION_REGISTRY_SHA256",
                        subject.file_hash(expansion))
    bindings = {
        "prior_v7_report_file_sha256": "1" * 64,
        "expansion_rows_file_sha256": "2" * 64,
        "fit_config_file_sha256": "3" * 64,
        "expansion_registry_file_sha256": subject.file_hash(expansion),
    }
    report = {
        "version": subject.audit_api.RCPD_VERSION,
        "status": "passed_development_gates", "explanation_eligible": True,
        "formal_ready": False, "bindings": bindings,
        "evidence_artifacts": {
            "program.json": subject.file_hash(files["program.json"]),
            "rows.npz": subject.file_hash(files["rows.npz"]),
        },
        "program_file_sha256": subject.file_hash(files["program.json"]),
    }
    files["report.json"].write_text(json.dumps(report), encoding="utf-8")
    calls = []

    def strict(directory, **kwargs):
        calls.append((Path(directory), kwargs))
        return report

    monkeypatch.setattr(subject.rcpd_api, "read_saved_report", strict)
    monkeypatch.setattr(subject, "producer_sources", lambda: {"source.py": "a" * 64})
    singles = {
        "actor": external["actor"], "protocol": external["protocol"],
        "manifest": external["manifest"], "designation": external["designation"],
        "program": files["program.json"], "rcpd_report": files["report.json"],
    }
    authenticated, sources, developments = subject._authenticate_candidate_after_claim(
        singles=singles, development_paths=[supplement, expansion],
        row_paths=[files["rows.npz"]], retired_paths=retired)
    assert authenticated == report and sources == {"source.py": "a" * 64}
    assert calls[0][0] == candidate
    assert calls[0][1]["require_passed"] is True
    assert calls[0][1]["refit"] is True
    assert calls[0][1]["previous_development_path"] == supplement
    assert developments[subject.holdout_api.DEVELOPMENT_EXPANSION_VERSION][0] == expansion


def _fake_run_setup(tmp_path, monkeypatch, audit_error):
    registry_root, anchor = _ledger(tmp_path, monkeypatch)
    files = {}
    for name in ("actor", "protocol", "program", "rcpd_report", "manifest",
                 "designation", "selected_scenes", "candidate_prior_v7_rows_npz"):
        path = tmp_path / (name + ".bin")
        path.write_bytes(name.encode())
        files[name] = path
    development = []
    for index, version in enumerate((
            subject.holdout_api.DEVELOPMENT_SUPPLEMENT_VERSION,
            subject.holdout_api.DEVELOPMENT_EXPANSION_VERSION)):
        path = tmp_path / f"development-{index}.json"
        path.write_text('{"version":"' + version + '"}', encoding="utf-8")
        development.append(path)
    rows = tmp_path / "rows.npz"; rows.write_bytes(b"rows")
    retired = []
    for index, version in enumerate(subject.holdout_api.EXTERNAL_RETIRED_VERSIONS):
        path = tmp_path / f"retired-{index}.json"
        path.write_text('{"version":"' + version + '"}', encoding="utf-8")
        retired.append(path)

    monkeypatch.setattr(subject, "_input_paths", lambda **kwargs: (
        files, development, [rows], retired))
    identity = subject._campaign_identity()
    frozen_sources = {"source.py": "a" * 64}
    monkeypatch.setattr(subject, "producer_sources", lambda: frozen_sources)
    monkeypatch.setattr(subject, "_preclaim_identity",
                        lambda *args: identity)
    calls = []

    def authenticate(**kwargs):
        key = subject.campaign_key()
        assert anchor.is_file()
        assert (registry_root / key / "attempt_started.json").is_file()
        calls.append("rcpd_strict_refit")
        developments = {
            subject.holdout_api.DEVELOPMENT_SUPPLEMENT_VERSION:
                (development[0], {"version": subject.holdout_api.DEVELOPMENT_SUPPLEMENT_VERSION}),
            subject.holdout_api.DEVELOPMENT_EXPANSION_VERSION:
                (development[1], {"version": subject.holdout_api.DEVELOPMENT_EXPANSION_VERSION}),
        }
        return {"status": "passed_development_gates"}, frozen_sources, developments

    monkeypatch.setattr(subject, "_authenticate_candidate_after_claim", authenticate)

    def holdout_build(**kwargs):
        key = subject.campaign_key()
        registry = registry_root / key
        assert (registry / "candidate_authenticated.json").is_file()
        assert kwargs["legacy_v3_rows_path"] == files["candidate_prior_v7_rows_npz"]
        assert kwargs["claim_receipt_path"] == registry / "attempt_started.json"
        calls.append("holdout")
        output = Path(kwargs["output"]); output.mkdir()
        (output / "v3_exclusion.json").write_text("{}", encoding="utf-8")
        (output / "holdout.json").write_text("{}", encoding="utf-8")
        (output / "report.json").write_text("{}", encoding="utf-8")
        common = {
            "campaign_key": key,
            "candidate_identity_sha256": subject.digest(identity),
            "attempt_started_sha256": subject.file_hash(
                registry / "attempt_started.json"),
        }
        (registry / "holdout_started.json").write_text(subject.canonical({
            **common, "status": "started_no_retry"}) + "\n", encoding="utf-8")
        (registry / "holdout_completed.json").write_text(subject.canonical({
            **common, "status": "completed_program_blind"}) + "\n",
            encoding="utf-8")
        return {"status": "passed_program_blind_registry"}

    monkeypatch.setattr(subject.holdout_api, "build", holdout_build)
    monkeypatch.setattr(subject.holdout_api, "read_saved_holdout",
                        lambda *args, **kwargs: calls.append("holdout_read") or {})

    def audit(**kwargs):
        key = subject.campaign_key()
        registry = registry_root / key
        assert (registry / "holdout_completed.json").is_file()
        calls.append("audit")
        if audit_error is not None:
            raise audit_error
        common = {
            "campaign_key": key,
            "candidate_identity_sha256": subject.digest(identity),
            "attempt_started_sha256": subject.file_hash(
                registry / "attempt_started.json"),
            "holdout_completed_sha256": subject.file_hash(
                registry / "holdout_completed.json"),
        }
        (registry / "audit_started.json").write_text(subject.canonical({
            **common, "status": "started_no_retry"}) + "\n", encoding="utf-8")
        (registry / "audit_completed.json").write_text(subject.canonical({
            **common, "status": "passed"}) + "\n", encoding="utf-8")
        output = Path(kwargs["output"]); output.mkdir()
        (output / "inputs.json").write_text("{}", encoding="utf-8")
        (output / "report.json").write_text("{}", encoding="utf-8")
        (output / "evidence.npz").write_bytes(b"evidence")
        return {"status": "passed", "bindings": {"bound": True}}

    monkeypatch.setattr(subject.audit_api, "audit", audit)
    return files, development, [rows], retired, identity, calls


def _run_args(files, development, rows, retired, output):
    return dict(
        actor_path=files["actor"], protocol_path=files["protocol"],
        program_path=files["program"], rcpd_report_path=files["rcpd_report"],
        manifest_path=files["manifest"], designation_path=files["designation"],
        selected_scenes_path=files["selected_scenes"],
        development_registry_paths=development,
        development_rows_paths=rows, retired_holdout_paths=retired,
        output=output, selection_salt_path=output.parent / "unread-private-salt",
    )


def test_failure_is_completed_as_burned_and_output_deletion_does_not_refund(
        tmp_path, monkeypatch):
    setup = _fake_run_setup(tmp_path, monkeypatch, RuntimeError("audit failed"))
    files, development, rows, retired, identity, calls = setup
    output = tmp_path / "out"
    result = subject.run_final_once(**_run_args(
        files, development, rows, retired, output))
    assert result["status"] == "burned_failed"
    assert calls[:3] == ["rcpd_strict_refit", "holdout", "holdout_read"]
    registry = Path(result["registry"])
    completion = subject._read_json(
        registry / "attempt_completed.json", "completion")
    assert completion["status"] == "burned_failed"
    assert completion["retry_allowed"] is False

    shutil.rmtree(output)
    with pytest.raises(ValueError, match="already_reserved_no_retry"):
        subject.run_final_once(**_run_args(
            files, development, rows, retired, tmp_path / "other-out"))


def test_interrupt_is_recorded_then_reraised(tmp_path, monkeypatch):
    setup = _fake_run_setup(tmp_path, monkeypatch, KeyboardInterrupt())
    files, development, rows, retired, identity, calls = setup
    with pytest.raises(KeyboardInterrupt):
        subject.run_final_once(**_run_args(
            files, development, rows, retired, tmp_path / "out"))
    registry = tmp_path / "external" / "ledger" / subject.campaign_key()
    completion = subject._read_json(
        registry / "attempt_completed.json", "completion")
    assert completion["status"] == "burned_interrupted"
    assert completion["retry_allowed"] is False


def test_success_orders_claim_refit_holdout_strict_audit_and_physical_replay(
        tmp_path, monkeypatch):
    setup = _fake_run_setup(tmp_path, monkeypatch, None)
    files, development, rows, retired, identity, calls = setup

    def read(*args, **kwargs):
        assert kwargs["require_passed"] is True
        calls.append("audit_strict_read")
        return {"status": "passed"}

    def replay(*args, **kwargs):
        assert kwargs["expected_bindings"] == {"bound": True}
        calls.append("physical_replay")
        return {"status": "passed"}

    monkeypatch.setattr(subject.audit_api, "read_saved_report", read)
    monkeypatch.setattr(subject.audit_api, "replay_saved_audit", replay)
    result = subject.run_final_once(**_run_args(
        files, development, rows, retired, tmp_path / "success"))
    assert result["status"] == "completed_passed"
    assert calls == ["rcpd_strict_refit", "holdout", "holdout_read", "audit",
                     "audit_strict_read", "physical_replay"]
    completion = subject._read_json(
        Path(result["registry"]) / "attempt_completed.json", "completion")
    assert completion["candidate_authentication_status"] == (
        "passed_strict_reader_and_refit")
    assert completion["development_authentication_refit"] is True
    assert completion["audit_status"] == "passed"
    assert completion["physical_replay_status"] == "passed"


def test_read_completion_binds_permanent_anchor_and_detects_tamper(
        tmp_path, monkeypatch):
    registry_root, anchor = _ledger(tmp_path, monkeypatch)
    identity = subject._campaign_identity()
    key, registry, started = subject._claim(identity, output=tmp_path / "out")
    completion = {
        "version": subject.VERSION, "key": key, "campaign_key": key,
        "candidate_identity_sha256": subject.digest(identity),
        "status": "burned_failed", "identity": identity,
        "attempt_started_sha256": subject.file_hash(
            registry / "attempt_started.json"),
        "permanent_anchor_sha256": subject.file_hash(anchor),
        "phase_receipts": {},
        "automatic_retry": False, "retry_allowed": False,
    }
    subject._write_exclusive(
        registry / "attempt_completed.json", subject._bytes(completion))
    expected = subject.file_hash(registry / "attempt_completed.json")
    assert subject.read_completion(
        registry, expected_completion_sha256=expected,
        expected_identity=identity)["status"] == "burned_failed"
    anchor.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="permanent record"):
        subject.read_completion(
            registry, expected_completion_sha256=expected,
            expected_identity=identity)
