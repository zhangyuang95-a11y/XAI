import json
from pathlib import Path
import pwd
import shutil

import pytest

from backend.training import warehouse_r41_diagnostic_final_once_v8 as subject


def _exception_module_material(error, module_names):
    """Collect scalar material reachable through protected-module tracebacks."""
    material = []
    seen = set()

    def visit(value):
        identity = id(value)
        if identity in seen:
            return
        if isinstance(value, (BaseException, dict, list, tuple, set, frozenset)):
            seen.add(identity)
        if isinstance(value, bytes):
            material.extend((repr(value), value.hex()))
        elif isinstance(value, (str, int)):
            material.append(str(value))
        elif isinstance(value, BaseException):
            material.append(str(value))
            visit(value.args)
            visit(value.__dict__)
            traceback = value.__traceback__
            while traceback is not None:
                frame = traceback.tb_frame
                if frame.f_globals.get("__name__") in module_names:
                    visit(dict(frame.f_locals))
                traceback = traceback.tb_next
            if value.__cause__ is not None:
                visit(value.__cause__)
            if value.__context__ is not None:
                visit(value.__context__)
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(key)
                visit(item)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                visit(item)

    visit(error)
    return "\n".join(material)


def _assert_private_material_absent(error, fragments):
    rendered = _exception_module_material(error, {
        subject.__name__, subject.holdout_api.__name__,
    })
    assert all(str(fragment) not in rendered for fragment in fragments), rendered


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
    assert "development_expansion_report_sha256" in contract["campaign_key_inputs"]
    assert "retired_identity_projection_file_sha256" in (
        contract["campaign_key_inputs"])
    assert "retired_identity_projection_report_sha256" in (
        contract["campaign_key_inputs"])
    assert contract["caller_selectable_ledger_or_anchor"] is False
    assert {"program", "rcpd_report", "producer_sources", "output"}.issubset(
        contract["campaign_key_excludes"])
    identity = subject._campaign_identity()
    assert identity == subject.holdout_api._campaign_identity()
    assert identity["development_expansion_registry_sha256"] == (
        subject.EXPECTED_EXPANSION_REGISTRY_SHA256)
    assert subject.EXPECTED_EXPANSION_REGISTRY_SHA256 == (
        subject.EXPECTED_FRESH_OUTER_REGISTRY_SHA256
    ) == "bf5346f9dd70ff02773b3335efd36018eae1b9abf4da1c035108a89ec5bcb923"
    assert identity["development_expansion_report_sha256"] == (
        subject.EXPECTED_FRESH_OUTER_REPORT_SHA256)
    assert identity["retired_identity_projection_file_sha256"] == (
        subject.EXPECTED_RETIRED_IDENTITY_PROJECTION_SHA256)
    assert identity["retired_identity_projection_report_sha256"] == (
        subject.EXPECTED_RETIRED_IDENTITY_PROJECTION_REPORT_SHA256)
    assert subject.campaign_key() == subject.digest(identity)
    sources = subject.producer_sources()
    assert {
        "backend/training/warehouse_r41_diagnostic_final_once_v8.py",
        "backend/training/warehouse_r41_diagnostic_fresh_final_holdout_v4.py",
        "backend/training/warehouse_r41_diagnostic_explanation_audit_v8.py",
        "backend/training/warehouse_r41_diagnostic_rcpd_v8.py",
        "scripts/build_warehouse_r41_diagnostic_designation_v2.py",
        "env/warehouse/transition_outcome.py",
    }.issubset(sources)
    assert "core/__init__.py" in sources


def test_placeholder_commitment_refuses_before_claim(monkeypatch):
    monkeypatch.setattr(subject.holdout_api, "HOLDOUT_SALT_COMMITMENT", "0" * 64)
    with pytest.raises(ValueError, match="not provisioned"):
        subject.campaign_key()


def test_preclaim_does_not_hash_program_or_report_and_authenticates_projection(
        tmp_path, monkeypatch):
    names = ("actor", "protocol", "program", "rcpd_report", "manifest",
             "designation", "selected_scenes", "fresh_outer_registry_report")
    singles = {name: tmp_path / name for name in names}
    developments = [tmp_path / "supplement", tmp_path / "expansion"]
    projection_paths = [tmp_path / "projection", tmp_path / "report.json"]
    implicit = {
        "manifest_validation": tmp_path / "validation",
        "designation_actor": tmp_path / "designation-actor",
        "designation_protocol": tmp_path / "designation-protocol",
        "designation_training_ledger": tmp_path / "designation-ledger",
        "designation_dual_evaluation": tmp_path / "designation-dual",
        "designation_failure_closeout": tmp_path / "designation-closeout",
    }
    expected = {
        singles["actor"]: subject.EXPECTED_ACTOR_SHA256,
        singles["protocol"]: subject.EXPECTED_PROTOCOL_SHA256,
        singles["manifest"]: subject.EXPECTED_MANIFEST_SHA256,
        singles["designation"]: subject.EXPECTED_DESIGNATION_SHA256,
        singles["selected_scenes"]: subject.EXPECTED_SELECTED_SCENES_SHA256,
        singles["fresh_outer_registry_report"]: (
            subject.EXPECTED_FRESH_OUTER_REPORT_SHA256),
        developments[0]: subject.EXPECTED_DEVELOPMENT_SUPPLEMENT_SHA256,
        developments[1]: subject.EXPECTED_EXPANSION_REGISTRY_SHA256,
        projection_paths[0]: subject.EXPECTED_RETIRED_IDENTITY_PROJECTION_SHA256,
        projection_paths[1]: (
            subject.EXPECTED_RETIRED_IDENTITY_PROJECTION_REPORT_SHA256),
        implicit["manifest_validation"]: (
            subject.manifest_binding.EXPECTED_VALIDATION_SHA256),
        implicit["designation_actor"]: subject.designation_api.EXPECTED_ACTOR_SHA256,
        implicit["designation_protocol"]: (
            subject.designation_api.EXPECTED_PROTOCOL_FILE_SHA256),
        implicit["designation_training_ledger"]: (
            subject.designation_api.EXPECTED_LEDGER_SHA256),
        implicit["designation_dual_evaluation"]: (
            subject.designation_api.EXPECTED_DUAL_EVALUATION_SHA256),
        implicit["designation_failure_closeout"]: (
            subject.designation_api.EXPECTED_CLOSEOUT_SHA256),
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
        subject.holdout_api, "_read_regular_bytes",
        lambda path, label, maximum=None: (b"", hashes(path)))
    monkeypatch.setattr(
        subject, "producer_sources",
        lambda: (_ for _ in ()).throw(AssertionError("sources accessed before claim")))
    binding = {"identity_count": 139}
    monkeypatch.setattr(
        subject.holdout_api, "_retired_identity_projection",
        lambda path: ([], binding))
    monkeypatch.setattr(
        subject.holdout_api, "_validate_retired_expansion_binding",
        lambda actual, expansion: actual == binding or (_ for _ in ()).throw(
            AssertionError("projection binding differs")))
    monkeypatch.setattr(
        subject.holdout_api, "_validate_fresh_outer_registry", lambda value: None)
    monkeypatch.setattr(
        subject.holdout_api, "_read_exact_json",
        lambda path, label, expected_sha256=None: ({}, expected_sha256))
    monkeypatch.setattr(
        subject.holdout_api, "_validate_fresh_outer_report",
        lambda report, registry: None)
    monkeypatch.setattr(
        subject.holdout_api, "_development_registry_evidence",
        lambda paths: (
            [],
            {
                subject.holdout_api.DEVELOPMENT_SUPPLEMENT_VERSION: {
                    "file_sha256": expected[developments[0]]},
                subject.holdout_api.DEVELOPMENT_EXPANSION_VERSION: {
                    "file_sha256": expected[developments[1]]},
            },
            {subject.holdout_api.DEVELOPMENT_EXPANSION_VERSION: {
                "version": subject.holdout_api.DEVELOPMENT_EXPANSION_VERSION,
                "bindings": {
                    "previous_development_file_sha256": (
                        subject.EXPECTED_DEVELOPMENT_SUPPLEMENT_SHA256),
                },
            }},
        ))
    assert subject._preclaim_identity(
        singles, developments, projection_paths, implicit
    ) == subject._campaign_identity()
    assert singles["program"] not in seen and singles["rcpd_report"] not in seen


def test_preclaim_rejects_wrong_supplement_before_program_access(
        tmp_path, monkeypatch):
    names = ("actor", "protocol", "program", "rcpd_report", "manifest",
             "designation", "selected_scenes")
    singles = {name: tmp_path / name for name in names}
    developments = [tmp_path / "supplement", tmp_path / "expansion"]
    projection_paths = [tmp_path / "projection", tmp_path / "report.json"]
    implicit = {
        "manifest_validation": tmp_path / "validation",
        "designation_actor": tmp_path / "designation-actor",
        "designation_protocol": tmp_path / "designation-protocol",
        "designation_training_ledger": tmp_path / "designation-ledger",
        "designation_dual_evaluation": tmp_path / "designation-dual",
        "designation_failure_closeout": tmp_path / "designation-closeout",
    }
    expected = {
        singles["actor"]: subject.EXPECTED_ACTOR_SHA256,
        singles["protocol"]: subject.EXPECTED_PROTOCOL_SHA256,
        singles["manifest"]: subject.EXPECTED_MANIFEST_SHA256,
        singles["designation"]: subject.EXPECTED_DESIGNATION_SHA256,
        singles["selected_scenes"]: subject.EXPECTED_SELECTED_SCENES_SHA256,
        projection_paths[0]: subject.EXPECTED_RETIRED_IDENTITY_PROJECTION_SHA256,
        projection_paths[1]: (
            subject.EXPECTED_RETIRED_IDENTITY_PROJECTION_REPORT_SHA256),
        implicit["manifest_validation"]: (
            subject.manifest_binding.EXPECTED_VALIDATION_SHA256),
        implicit["designation_actor"]: subject.designation_api.EXPECTED_ACTOR_SHA256,
        implicit["designation_protocol"]: (
            subject.designation_api.EXPECTED_PROTOCOL_FILE_SHA256),
        implicit["designation_training_ledger"]: (
            subject.designation_api.EXPECTED_LEDGER_SHA256),
        implicit["designation_dual_evaluation"]: (
            subject.designation_api.EXPECTED_DUAL_EVALUATION_SHA256),
        implicit["designation_failure_closeout"]: (
            subject.designation_api.EXPECTED_CLOSEOUT_SHA256),
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
        subject.holdout_api, "_read_regular_bytes",
        lambda path, label, maximum=None: (b"", hashes(path)))
    monkeypatch.setattr(
        subject.holdout_api, "_retired_identity_projection",
        lambda path: ([], {"identity_count": 139}))
    monkeypatch.setattr(
        subject.holdout_api, "_development_registry_evidence",
        lambda paths: (
            [],
            {
                subject.holdout_api.DEVELOPMENT_SUPPLEMENT_VERSION: {
                    "file_sha256": "0" * 64},
                subject.holdout_api.DEVELOPMENT_EXPANSION_VERSION: {
                    "file_sha256": subject.EXPECTED_EXPANSION_REGISTRY_SHA256},
            },
            {},
        ))
    with pytest.raises(ValueError, match="development supplement differs"):
        subject._preclaim_identity(
            singles, developments, projection_paths, implicit)
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


def test_two_caller_environment_paths_cannot_create_two_campaign_claims(
        tmp_path, monkeypatch):
    registry_root, anchor = _ledger(tmp_path, monkeypatch)
    identity = subject._campaign_identity()
    attacker_ledger_a = tmp_path / "attacker-a" / "ledger"
    attacker_anchor_a = tmp_path / "attacker-a" / "anchor"
    monkeypatch.setenv(subject.LEDGER_ENV, str(attacker_ledger_a))
    monkeypatch.setenv(subject.PERMANENT_ANCHOR_ENV, str(attacker_anchor_a))
    monkeypatch.setenv("HOME", str(tmp_path / "attacker-a" / "home"))
    _, registry, _ = subject._claim(identity, output=tmp_path / "out-a")
    assert registry == registry_root / subject.campaign_key()
    assert anchor.is_file()
    assert not attacker_ledger_a.exists() and not attacker_anchor_a.exists()

    attacker_ledger_b = tmp_path / "attacker-b" / "ledger"
    attacker_anchor_b = tmp_path / "attacker-b" / "anchor"
    monkeypatch.setenv(subject.LEDGER_ENV, str(attacker_ledger_b))
    monkeypatch.setenv(subject.PERMANENT_ANCHOR_ENV, str(attacker_anchor_b))
    monkeypatch.setenv("HOME", str(tmp_path / "attacker-b" / "home"))
    with pytest.raises(ValueError, match="already_reserved_no_retry"):
        subject._claim(identity, output=tmp_path / "out-b")
    assert not attacker_ledger_b.exists() and not attacker_anchor_b.exists()


def test_default_registry_anchor_and_salt_use_account_home_not_home_env(
        tmp_path, monkeypatch):
    account_home = Path(pwd.getpwuid(subject.os.getuid()).pw_dir).resolve()
    monkeypatch.setenv("HOME", str(tmp_path / "attacker-home"))
    assert subject.holdout_api.DEFAULT_LEDGER_ROOT.is_relative_to(account_home)
    assert subject.holdout_api.DEFAULT_PERMANENT_ANCHOR.is_relative_to(account_home)
    assert subject.DEFAULT_SALT_FILE.is_relative_to(account_home)


def test_input_paths_accept_only_projection_pair_and_one_candidate_directory(tmp_path):
    candidate = tmp_path / "candidate"; candidate.mkdir()
    paths = {}
    for name in subject.CANDIDATE_ARTIFACT_KEYS:
        path = candidate / name; path.write_bytes(name.encode()); paths[name] = path
    singles = {}
    for name in ("actor", "protocol", "manifest", "designation", "selected"):
        path = tmp_path / name; path.write_bytes(name.encode()); singles[name] = path
    dev = []
    for index in range(2):
        path = tmp_path / f"dev-{index}"; path.write_bytes(b"dev"); dev.append(path)
    projection_dir = tmp_path / "projection"; projection_dir.mkdir()
    projection = projection_dir / "retired_identity_projection.json"
    projection.write_bytes(b"projection")
    (projection_dir / "report.json").write_bytes(b"projection report")
    kwargs = dict(
        actor_path=singles["actor"], protocol_path=singles["protocol"],
        program_path=paths["program.json"], rcpd_report_path=paths["report.json"],
        manifest_path=singles["manifest"], designation_path=singles["designation"],
        selected_scenes_path=singles["selected"], development_registry_paths=dev,
        development_rows_paths=[paths["rows.npz"]],
        retired_identity_projection_path=projection,
    )
    result = subject._input_paths(**kwargs)
    assert len(result[2]) == 1 and len(result[3]) == 2
    (projection_dir / "report.json").unlink()
    with pytest.raises(ValueError, match="projection report"):
        subject._input_paths(**kwargs)


def test_fresh_outer_report_is_a_fixed_sibling_input(tmp_path, monkeypatch):
    supplement = tmp_path / "supplement.json"
    supplement.write_text('{"kind":"supplement"}\n', encoding="utf-8")
    outer = tmp_path / "outer"; outer.mkdir()
    registry = outer / "development_expansion.json"
    registry.write_text("{}\n", encoding="utf-8")
    report = outer / "report.json"
    report.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        subject, "EXPECTED_FRESH_OUTER_REGISTRY_SHA256",
        subject.file_hash(registry))
    monkeypatch.setattr(
        subject, "EXPECTED_FRESH_OUTER_REPORT_SHA256",
        subject.file_hash(report))

    assert subject._fresh_outer_report_input([supplement, registry]) == report
    report.write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="registry report"):
        subject._fresh_outer_report_input([supplement, registry])


def test_candidate_selector_artifacts_are_individually_frozen(tmp_path):
    candidate = tmp_path / "candidate"; candidate.mkdir()
    files = {}
    singles = {}
    for name, key in subject.CANDIDATE_ARTIFACT_KEYS.items():
        path = candidate / name
        path.write_bytes(name.encode())
        files[name] = path
        if key != "development_rows":
            singles[key] = path
    rows = [files["rows.npz"]]
    artifacts = subject._candidate_artifact_hashes(singles, rows)
    marker = {
        "candidate_artifacts": artifacts,
        "candidate_artifacts_sha256": subject.digest(artifacts),
        "program_file_sha256": artifacts["program.json"],
        "rcpd_report_file_sha256": artifacts["report.json"],
        "rows_file_sha256": artifacts["rows.npz"],
        "prior_rows_reauthentication_report_file_sha256": artifacts[
            "prior_rows_reauthentication_report.json"],
        "prior_v7_source_report_file_sha256": artifacts[
            "source_v7_report.json"],
        "prior_v7_rows_file_sha256": artifacts["prior_v7_rows.npz"],
        "expansion_rows_reauthentication_report_file_sha256": artifacts[
            "expansion_rows_reauthentication_report.json"],
        "expansion_source_collection_report_file_sha256": artifacts[
            "source_expansion_collection_report.json"],
        "expansion_rows_file_sha256": artifacts["expansion_rows.npz"],
        "development_expansion_registry_file_sha256": artifacts[
            "development_expansion.json"],
        "development_expansion_report_file_sha256": artifacts[
            "development_expansion_report.json"],
        "fit_config_file_sha256": artifacts["fit_config.json"],
        "fit_selector_report_file_sha256": artifacts[
            "fit_selector_report.json"],
        "fit_selector_scope_file_sha256": artifacts[
            "fit_selector_scope.json"],
        "fit_selector_selected_config_file_sha256": artifacts[
            "fit_selector_selected_config.json"],
    }
    assert subject._validate_candidate_artifact_binding(
        marker, singles=singles, row_paths=rows) == artifacts
    for field in (
        "fit_selector_report_file_sha256",
        "fit_selector_scope_file_sha256",
        "fit_selector_selected_config_file_sha256",
    ):
        changed = dict(marker)
        changed[field] = "f" * 64
        with pytest.raises(ValueError, match="candidate artifacts changed"):
            subject._validate_candidate_artifact_binding(
                changed, singles=singles, row_paths=rows)


def test_candidate_authentication_uses_same_directory_strict_refit(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate"; candidate.mkdir()
    files = {}
    for name in subject.CANDIDATE_ARTIFACT_KEYS:
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
    projection_dir = tmp_path / "projection"; projection_dir.mkdir()
    projection = projection_dir / "retired_identity_projection.json"
    projection.write_text("{}", encoding="utf-8")
    projection_report = projection_dir / "report.json"
    projection_report.write_text("{}", encoding="utf-8")
    projection_paths = [projection, projection_report]
    monkeypatch.setattr(
        subject, "EXPECTED_RETIRED_IDENTITY_PROJECTION_SHA256",
        subject.file_hash(projection))
    monkeypatch.setattr(
        subject, "EXPECTED_RETIRED_IDENTITY_PROJECTION_REPORT_SHA256",
        subject.file_hash(projection_report))
    projection_binding = {
        "file_sha256": subject.file_hash(projection),
        "audit_file_sha256": subject.file_hash(projection_report),
        "content_sha256": "1" * 64, "identity_count": 139,
    }
    monkeypatch.setattr(
        subject.holdout_api, "_retired_identity_projection",
        lambda path: ([], projection_binding))
    monkeypatch.setattr(
        subject.holdout_api, "_validate_retired_expansion_binding",
        lambda actual, expansion: None if actual == projection_binding
        else (_ for _ in ()).throw(
            ValueError("projection differs")))
    expansion = tmp_path / "expansion.json"
    expansion.write_text(json.dumps({
        "version": subject.holdout_api.DEVELOPMENT_EXPANSION_VERSION,
        "bindings": {"retired_identity_projection": projection_binding},
    }), encoding="utf-8")
    monkeypatch.setattr(subject, "EXPECTED_EXPANSION_REGISTRY_SHA256",
                        subject.file_hash(expansion))
    files["development_expansion_report.json"].write_text(
        "{}", encoding="utf-8")
    files["development_expansion_report.json"].write_text("{}\n", encoding="utf-8")
    outer_report = tmp_path / "report.json"
    outer_report.write_bytes(files["development_expansion_report.json"].read_bytes())
    monkeypatch.setattr(
        subject, "EXPECTED_FRESH_OUTER_REPORT_SHA256",
        subject.file_hash(outer_report))
    monkeypatch.setattr(
        subject.holdout_api, "_validate_fresh_outer_registry",
        lambda value: None)
    monkeypatch.setattr(
        subject.holdout_api, "_validate_fresh_outer_report",
        lambda report, registry: None)
    bindings = {
        "prior_rows_reauthentication_receipt_file_sha256": subject.file_hash(
            files["prior_rows_reauthentication_report.json"]),
        "prior_v7_source_report_file_sha256": subject.file_hash(
            files["source_v7_report.json"]),
        "prior_v7_rows_file_sha256": subject.file_hash(
            files["prior_v7_rows.npz"]),
        "expansion_rows_reauthentication_receipt_file_sha256": subject.file_hash(
            files["expansion_rows_reauthentication_report.json"]),
        "expansion_rows_file_sha256": subject.file_hash(files["expansion_rows.npz"]),
        "fit_config_file_sha256": subject.file_hash(files["fit_config.json"]),
        "expansion_registry_file_sha256": subject.file_hash(expansion),
        "expansion_registry_report_file_sha256": subject.file_hash(
            files["development_expansion_report.json"]),
        "previous_development_file_sha256": subject.file_hash(supplement),
        "fit_selector_report_file_sha256": subject.file_hash(
            files["fit_selector_report.json"]),
        "fit_selector_scope_file_sha256": subject.file_hash(
            files["fit_selector_scope.json"]),
        "fit_selector_selected_config_file_sha256": subject.file_hash(
            files["fit_selector_selected_config.json"]),
    }
    report = {
        "version": subject.audit_api.RCPD_VERSION,
        "status": "passed_development_gates", "explanation_eligible": True,
        "formal_ready": False, "bindings": bindings,
        "evidence_artifacts": {
            name: subject.file_hash(files[name])
            for name in subject.CANDIDATE_ARTIFACT_KEYS
            if name != "report.json"
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
        "fresh_outer_registry_report": outer_report,
    }
    for name, key in subject.CANDIDATE_ARTIFACT_KEYS.items():
        if key not in {"program", "rcpd_report", "development_rows"}:
            singles[key] = files[name]
    authenticated, sources, developments, candidate_artifacts = (
        subject._authenticate_candidate_after_claim(
        singles=singles, development_paths=[supplement, expansion],
        row_paths=[files["rows.npz"]], projection_paths=projection_paths))
    assert authenticated == report and sources == {"source.py": "a" * 64}
    assert candidate_artifacts == {
        name: subject.file_hash(candidate / name)
        for name in sorted(subject.CANDIDATE_ARTIFACT_KEYS)
    }
    assert calls[0][0] == candidate
    assert calls[0][1]["require_passed"] is True
    assert calls[0][1]["refit"] is True
    assert calls[0][1]["previous_development_path"] == supplement
    assert calls[0][1]["expected_prior_rows_report_sha256"] == subject.file_hash(
        files["prior_rows_reauthentication_report.json"])
    assert calls[0][1]["expected_expansion_rows_report_sha256"] == subject.file_hash(
        files["expansion_rows_reauthentication_report.json"])
    assert calls[0][1]["expected_selector_report_sha256"] == subject.file_hash(
        files["fit_selector_report.json"])
    assert "expected_config_sha256" not in calls[0][1]
    assert developments[subject.holdout_api.DEVELOPMENT_EXPANSION_VERSION][0] == expansion


def _fake_run_setup(tmp_path, monkeypatch, audit_error):
    registry_root, anchor = _ledger(tmp_path, monkeypatch)
    files = {}
    file_keys = {
        "actor", "protocol", "program", "rcpd_report", "manifest",
        "designation", "selected_scenes",
        *(key for key in subject.CANDIDATE_ARTIFACT_KEYS.values()
          if key not in {"program", "rcpd_report", "development_rows"}),
    }
    for name in sorted(file_keys):
        path = tmp_path / (name + ".bin")
        path.write_bytes(name.encode())
        files[name] = path
    for constant, name in (
            ("EXPECTED_ACTOR_SHA256", "actor"),
            ("EXPECTED_PROTOCOL_SHA256", "protocol"),
            ("EXPECTED_MANIFEST_SHA256", "manifest"),
            ("EXPECTED_DESIGNATION_SHA256", "designation"),
            ("EXPECTED_SELECTED_SCENES_SHA256", "selected_scenes")):
        monkeypatch.setattr(subject, constant, subject.file_hash(files[name]))
    development = []
    for index, version in enumerate((
            subject.holdout_api.DEVELOPMENT_SUPPLEMENT_VERSION,
            subject.holdout_api.DEVELOPMENT_EXPANSION_VERSION)):
        path = tmp_path / f"development-{index}.json"
        path.write_text('{"version":"' + version + '"}', encoding="utf-8")
        development.append(path)
    monkeypatch.setattr(
        subject, "EXPECTED_EXPANSION_REGISTRY_SHA256",
        subject.file_hash(development[1]))
    outer_report = tmp_path / "fresh-outer-report.json"
    outer_report.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        subject, "EXPECTED_FRESH_OUTER_REPORT_SHA256",
        subject.file_hash(outer_report))
    monkeypatch.setattr(
        subject, "_fresh_outer_report_input", lambda paths: outer_report)
    rows = tmp_path / "rows.npz"; rows.write_bytes(b"rows")
    implicit_paths = {}
    implicit_constants = {
        "manifest_validation": (subject.manifest_binding,
                                "EXPECTED_VALIDATION_SHA256"),
        "designation_actor": (subject.designation_api, "EXPECTED_ACTOR_SHA256"),
        "designation_protocol": (subject.designation_api,
                                 "EXPECTED_PROTOCOL_FILE_SHA256"),
        "designation_training_ledger": (subject.designation_api,
                                        "EXPECTED_LEDGER_SHA256"),
        "designation_dual_evaluation": (subject.designation_api,
                                        "EXPECTED_DUAL_EVALUATION_SHA256"),
        "designation_failure_closeout": (subject.designation_api,
                                         "EXPECTED_CLOSEOUT_SHA256"),
    }
    for name, (module, constant) in implicit_constants.items():
        path = tmp_path / (
            "implicit-validation.json" if name == "manifest_validation"
            else name + ".bin")
        path.write_bytes(name.encode())
        implicit_paths[name] = path
        monkeypatch.setattr(module, constant, subject.file_hash(path))
    projection_dir = tmp_path / "projection"; projection_dir.mkdir()
    projection = projection_dir / "retired_identity_projection.json"
    projection.write_text("{}", encoding="utf-8")
    projection_report = projection_dir / "report.json"
    projection_report.write_text("{}", encoding="utf-8")
    projection_paths = [projection, projection_report]

    monkeypatch.setattr(subject, "_input_paths", lambda **kwargs: (
        files, development, [rows], projection_paths))
    monkeypatch.setattr(
        subject, "_implicit_input_paths",
        lambda singles: implicit_paths)
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
        candidate_artifacts = subject._candidate_artifact_hashes(files, [rows])
        return ({
            "status": "passed_development_gates",
            "bindings": {
                "previous_development_file_sha256": subject.file_hash(
                    development[0]),
                "expansion_registry_file_sha256": subject.file_hash(
                    development[1]),
            },
        }, frozen_sources,
                developments, candidate_artifacts)

    monkeypatch.setattr(subject, "_authenticate_candidate_after_claim", authenticate)

    def holdout_build(**kwargs):
        key = subject.campaign_key()
        registry = registry_root / key
        assert (registry / "candidate_authenticated.json").is_file()
        assert kwargs["legacy_v3_rows_path"] == files["candidate_prior_v7_rows_npz"]
        assert kwargs["retired_identity_projection_path"] == projection
        assert kwargs["claim_receipt_path"] == registry / "attempt_started.json"
        calls.append("holdout")
        output = Path(kwargs["output"]); output.mkdir()
        (output / "v3_exclusion.json").write_text("{}", encoding="utf-8")
        (output / "holdout.json").write_text("{}", encoding="utf-8")
        holdout_report = {"status": "passed_program_blind_registry"}
        (output / "report.json").write_text(
            subject.canonical(holdout_report) + "\n", encoding="utf-8")
        common = {
            "version": subject.audit_api.VERSION,
            "campaign_key": key,
            "candidate_identity_sha256": subject.digest(identity),
            "attempt_started_sha256": subject.file_hash(
                registry / "attempt_started.json"),
            "candidate_authenticated_sha256": subject.file_hash(
                registry / "candidate_authenticated.json"),
        }
        holdout_started = {
            **common, "version": subject.holdout_api.VERSION,
            "status": "started_no_retry",
            "selection_salt_commitment": (
                subject.holdout_api.HOLDOUT_SALT_COMMITMENT),
        }
        (registry / "holdout_started.json").write_text(
            subject.canonical(holdout_started) + "\n", encoding="utf-8")
        candidate = subject._read_json(
            registry / "candidate_authenticated.json", "candidate marker")
        rows_by_name = {
            "merged_rows.npz": candidate["candidate_artifacts"]["rows.npz"],
            "prior_rows.npz": candidate["candidate_artifacts"]["prior_v7_rows.npz"],
            "expansion_rows.npz": candidate["candidate_artifacts"]["expansion_rows.npz"],
        }
        started_rows = {
            name: {
                "file_sha256": file_sha, "row_count": index + 1,
                "unique_scene_fingerprint_count": index + 2,
                "scene_fingerprints_sha256": str(index + 1) * 64,
                "unique_public_observation_count": index + 3,
                "public_observations_sha256": str(index + 4) * 64,
            }
            for index, (name, file_sha) in enumerate(sorted(rows_by_name.items()))
        }
        historical_started = {
            **common,
            "version": subject.holdout_api.VERSION + ".historical-exclusion.v1",
            "status": "started_irrevocable_no_retry",
            "holdout_started_sha256": subject.file_hash(
                registry / "holdout_started.json"),
            "manifest_file_sha256": subject.EXPECTED_MANIFEST_SHA256,
            "row_artifact_evidence": started_rows,
            "row_artifact_evidence_sha256": subject.digest(started_rows),
            "public_exclusion_commitment": {
                "scene_fingerprint_count": 1,
                "scene_fingerprints_sha256": "1" * 64,
                "seed_count": 1,
                "seeds_sha256": "2" * 64,
                "replayed_observation_count": 1,
                "replayed_observations_sha256": "3" * 64,
                "forbidden_observation_count": 1,
                "forbidden_observations_sha256": "4" * 64,
                "retired_identity_projection_file_sha256": "5" * 64,
                "retired_identity_projection_report_sha256": "6" * 64,
                "retired_identity_count": 139,
                "retired_actor_executed": False,
                "retired_observations_derived": False,
                "v3_exclusion_content_sha256": "7" * 64,
            },
            "historical_final_access_refunds_attempt": False,
            "formal_ready": False,
        }
        historical_started["public_exclusion_commitment_sha256"] = subject.digest(
            historical_started["public_exclusion_commitment"])
        (registry / "historical_exclusion_started.json").write_text(
            subject.canonical(historical_started) + "\n", encoding="utf-8")
        completed_rows = {
            name: {
                **value,
                "historical_scene_fingerprint_overlap": 0,
                "historical_public_observation_overlap": 0,
                "zero_historical_overlap": True,
            }
            for name, value in started_rows.items()
        }
        historical_completed = {
            **common,
            "version": subject.holdout_api.VERSION + ".historical-exclusion.v1",
            "status": "completed_observation_hash_exclusion",
            "holdout_started_sha256": subject.file_hash(
                registry / "holdout_started.json"),
            "historical_exclusion_started_sha256": subject.file_hash(
                registry / "historical_exclusion_started.json"),
            "row_artifact_evidence": completed_rows,
            "row_artifact_evidence_sha256": subject.digest(completed_rows),
            "historical_final_scenes_returned": False,
            "historical_final_actor_outputs_exposed": False,
            "historical_final_actor_outputs_persisted": False,
            "historical_final_labels_used": False,
            "historical_final_saved_metrics_replayed_for_authentication": False,
            "historical_final_metrics_exposed": False,
            "historical_final_metrics_persisted": False,
            "historical_final_metrics_used_for_fit_or_program_selection": False,
            "historical_final_used_for_fit_or_program_selection": False,
            "historical_final_replayed_for_exclusion_only": True,
            "historical_final_actor_executed_for_observation_exclusion_only": True,
            "fresh_selection_conditioned_on_historical_final": False,
            "fresh_selection_private_overlap_fallback": False,
            "fresh_selection_historical_scene_fingerprint_overlap": 0,
            "fresh_selection_historical_seed_overlap": 0,
            "fresh_selection_historical_public_observation_overlap": 0,
            "manifest_file_sha256": subject.EXPECTED_MANIFEST_SHA256,
            "historical_final_identity_sha256": (
                subject.holdout_api.manifest_binding.EXPECTED_FINAL_IDENTITY_SHA256),
            "historical_final_scene_count": subject.holdout_api.TOTAL_SCENES,
            "historical_final_public_observation_count": 1,
            "historical_final_public_observations_sha256": "8" * 64,
            "combined_excluded_scene_fingerprint_count": (
                subject.holdout_api.TOTAL_SCENES + 1),
            "combined_excluded_scene_fingerprints_sha256": "9" * 64,
            "combined_excluded_seed_count": subject.holdout_api.TOTAL_SCENES + 1,
            "combined_excluded_seeds_sha256": "a" * 64,
            "combined_forbidden_public_observation_count": 2,
            "combined_forbidden_public_observations_sha256": "b" * 64,
            "combined_replayed_excluded_public_observation_count": 2,
            "combined_replayed_excluded_public_observations_sha256": "c" * 64,
            "development_row_scene_overlap": 0,
            "preexisting_scene_identity_overlap": 0,
            "preexisting_public_observation_overlap": 0,
            "retry_allowed": False,
            "formal_ready": False,
        }
        historical_completed["receipt_sha256"] = subject.digest(
            historical_completed)
        (registry / "historical_exclusion_completed.json").write_text(
            subject.canonical(historical_completed) + "\n", encoding="utf-8")
        holdout_completed = {
            **common, "version": subject.holdout_api.VERSION,
            "status": "completed_program_blind",
            "historical_exclusion_completed_sha256": subject.file_hash(
                registry / "historical_exclusion_completed.json"),
            "holdout_file_sha256": subject.file_hash(output / "holdout.json"),
            "report_file_sha256": subject.file_hash(output / "report.json"),
            "v3_exclusion_file_sha256": subject.file_hash(
                output / "v3_exclusion.json"),
        }
        (registry / "holdout_completed.json").write_text(
            subject.canonical(holdout_completed) + "\n", encoding="utf-8")
        return {
            **holdout_report,
            "_publication_attestation": {
                "artifacts": {
                    name: subject.file_hash(output / name)
                    for name in ("v3_exclusion.json", "holdout.json", "report.json")
                },
                "holdout_completed_sha256": subject.file_hash(
                    registry / "holdout_completed.json"),
                "historical_exclusion_started_sha256": subject.file_hash(
                    registry / "historical_exclusion_started.json"),
                "historical_exclusion_completed_sha256": subject.file_hash(
                    registry / "historical_exclusion_completed.json"),
            },
        }

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
            "version": subject.audit_api.VERSION,
            "campaign_key": key,
            "candidate_identity_sha256": subject.digest(identity),
            "attempt_started_sha256": subject.file_hash(
                registry / "attempt_started.json"),
            "holdout_completed_sha256": subject.file_hash(
                registry / "holdout_completed.json"),
            "candidate_authenticated_sha256": subject.file_hash(
                registry / "candidate_authenticated.json"),
            "candidate_artifacts_sha256": subject.digest(
                subject._candidate_artifact_hashes(files, [rows])),
        }
        output = Path(kwargs["output"]); output.mkdir()
        (output / "inputs.json").write_text("{}", encoding="utf-8")
        (output / "evidence.npz").write_bytes(b"evidence")
        report = {
            "status": "passed", "bindings": {"bound": True},
            "evidence_artifacts": {
                "inputs.json": subject.file_hash(output / "inputs.json"),
                "evidence.npz": subject.file_hash(output / "evidence.npz"),
            },
        }
        (output / "report.json").write_text(
            subject.canonical(report) + "\n", encoding="utf-8")
        (registry / "audit_started.json").write_text(subject.canonical({
            **common, "status": "started_no_retry"}) + "\n", encoding="utf-8")
        (registry / "audit_completed.json").write_text(subject.canonical({
            **common, "status": "passed",
            "report_file_sha256": subject.file_hash(output / "report.json"),
            "evidence_file_sha256": subject.file_hash(output / "evidence.npz"),
        }) + "\n", encoding="utf-8")
        return report

    monkeypatch.setattr(subject.audit_api, "audit", audit)
    return files, development, [rows], projection_paths, identity, calls


def _run_args(files, development, rows, projection_paths, output):
    return dict(
        actor_path=files["actor"], protocol_path=files["protocol"],
        program_path=files["program"], rcpd_report_path=files["rcpd_report"],
        manifest_path=files["manifest"], designation_path=files["designation"],
        selected_scenes_path=files["selected_scenes"],
        development_registry_paths=development,
        development_rows_paths=rows,
        retired_identity_projection_path=projection_paths[0],
        output=output, selection_salt_path=output.parent / "unread-private-salt",
    )


@pytest.mark.parametrize("stale_schema", [False, True])
def test_candidate_interface_mismatch_is_rejected_before_permanent_claim(
        tmp_path, monkeypatch, stale_schema):
    setup = _fake_run_setup(tmp_path, monkeypatch, None)
    files, development, rows, retired, _, calls = setup
    if stale_schema:
        monkeypatch.setattr(
            subject.rcpd_api, "_CANDIDATE_SELECTOR_ARTIFACTS",
            frozenset({"fit_selector_report.json"}))
    else:
        def stale_reader(
            output, *, expected_report_sha256, actor_path, protocol_path,
            manifest_path, designation_path, expansion_registry_path,
            expected_expansion_registry_sha256, expansion_report_path,
            expected_expansion_report_sha256,
            expected_prior_rows_report_sha256, previous_development_path,
            expected_expansion_rows_report_sha256,
            expected_config_sha256, require_passed=True, refit=True,
        ):
            raise AssertionError("stale reader must not run")

        monkeypatch.setattr(subject.rcpd_api, "read_saved_report", stale_reader)
    with pytest.raises(RuntimeError, match="incompatible"):
        subject.run_final_once(**_run_args(
            files, development, rows, retired, tmp_path / "never-claimed"))
    assert calls == []
    assert not (tmp_path / "permanent" / "campaign.anchor").exists()
    assert not (tmp_path / "external" / "ledger").exists()


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
    assert completion["reason"] == "historical_final_private_phase_failed"
    assert "audit failed" not in completion["reason"]

    shutil.rmtree(output)
    with pytest.raises(ValueError, match="already_reserved_no_retry"):
        subject.run_final_once(**_run_args(
            files, development, rows, retired, tmp_path / "other-out"))


def test_private_failure_payload_never_reaches_durable_completion(
        tmp_path, monkeypatch):
    setup = _fake_run_setup(tmp_path, monkeypatch, None)
    files, development, rows, retired, _, _ = setup
    completed_holdout = subject.holdout_api.build
    private_fragments = (
        "private-fingerprint-7f4d", "private-seed-918273",
        "private-observation-c0ffee", "actor_action=RIGHT",
    )

    def fail_after_private_phase(**kwargs):
        completed_holdout(**kwargs)
        raise RuntimeError(" | ".join(private_fragments))

    monkeypatch.setattr(subject.holdout_api, "build", fail_after_private_phase)
    result = subject.run_final_once(**_run_args(
        files, development, rows, retired, tmp_path / "private-error"))
    assert result["status"] == "burned_failed"
    assert result["reason"] == "historical_final_private_phase_failed"
    registry = Path(result["registry"])
    completion_raw = (registry / "attempt_completed.json").read_text(
        encoding="utf-8")
    completion = subject._read_json(
        registry / "attempt_completed.json", "completion")
    assert completion["reason"] == "historical_final_private_phase_failed"
    assert all(fragment not in completion_raw for fragment in private_fragments)
    with pytest.raises(ValueError, match="already_reserved_no_retry"):
        subject.run_final_once(**_run_args(
            files, development, rows, retired, tmp_path / "private-error-retry"))


def test_post_salt_pre_historical_failure_is_durably_redacted_and_burned(
        tmp_path, monkeypatch):
    setup = _fake_run_setup(tmp_path, monkeypatch, None)
    files, development, rows, retired, identity, _ = setup
    private_fragments = (
        "private-salt-0123456789abcdef", "private-seed-918273",
        "private-fingerprint-7f4d", "private-observation-c0ffee",
    )
    payload = " | ".join(private_fragments)
    registry_root = tmp_path / "external" / "ledger"

    def fail_after_salt_before_historical(**kwargs):
        registry = registry_root / subject.campaign_key()
        common = {
            "campaign_key": subject.campaign_key(),
            "candidate_identity_sha256": subject.digest(identity),
            "attempt_started_sha256": subject.file_hash(
                registry / "attempt_started.json"),
            "candidate_authenticated_sha256": subject.file_hash(
                registry / "candidate_authenticated.json"),
        }
        (registry / "holdout_started.json").write_text(subject.canonical({
            **common, "status": "started_no_retry",
        }) + "\n", encoding="utf-8")
        raise RuntimeError(payload)

    monkeypatch.setattr(subject.holdout_api, "build",
                        fail_after_salt_before_historical)
    result = subject.run_final_once(**_run_args(
        files, development, rows, retired, tmp_path / "post-salt-failure"))
    assert result["status"] == "burned_failed"
    assert result["reason"] == "historical_final_private_phase_failed"
    registry = Path(result["registry"])
    completion_raw = (registry / "attempt_completed.json").read_text(
        encoding="utf-8")
    assert all(fragment not in completion_raw for fragment in private_fragments)
    with pytest.raises(ValueError, match="already_reserved_no_retry"):
        subject._claim(identity, output=tmp_path / "post-salt-retry")


def test_interrupt_is_recorded_then_reraised(tmp_path, monkeypatch):
    private_fragments = (
        "private-salt-abcdef0123456789", "private-seed-112358",
        "private-fingerprint-a11ce", "private-observation-deadbeef",
    )

    class PrivateInterrupt(BaseException):
        pass

    setup = _fake_run_setup(
        tmp_path, monkeypatch, PrivateInterrupt(" | ".join(private_fragments)))
    files, development, rows, retired, identity, calls = setup
    with pytest.raises(subject.holdout_api._HistoricalFinalPrivatePhaseInterrupt,
                       match="^historical_final_private_phase_interrupted$") as raised:
        subject.run_final_once(**_run_args(
            files, development, rows, retired, tmp_path / "out"))
    _assert_private_material_absent(raised.value, private_fragments)
    registry = tmp_path / "external" / "ledger" / subject.campaign_key()
    completion_raw = (registry / "attempt_completed.json").read_text(
        encoding="utf-8")
    assert all(fragment not in completion_raw for fragment in private_fragments)
    completion = subject._read_json(
        registry / "attempt_completed.json", "completion")
    assert completion["status"] == "burned_interrupted"
    assert completion["retry_allowed"] is False
    assert completion["reason"] == "historical_final_private_phase_interrupted"
    with pytest.raises(ValueError, match="already_reserved_no_retry"):
        subject._claim(identity, output=tmp_path / "interrupt-retry")


@pytest.mark.parametrize(
    "secondary_point",
    ["phase_receipts", "completion_serialization", "completion_publication"],
)
def test_primary_private_failure_is_not_retained_by_secondary_failure(
        tmp_path, monkeypatch, secondary_point):
    private_fragments = (
        "private-salt-fedcba9876543210", "private-seed-271828",
        "private-fingerprint-badc0de", "private-observation-facefeed",
    )
    setup = _fake_run_setup(
        tmp_path, monkeypatch, RuntimeError(" | ".join(private_fragments)))
    files, development, rows, retired, identity, _ = setup

    if secondary_point == "phase_receipts":
        monkeypatch.setattr(
            subject, "_phase_receipt_hashes",
            lambda _registry: (_ for _ in ()).throw(
                OSError("secondary phase-receipt failure")))
    elif secondary_point == "completion_serialization":
        original_bytes = subject._bytes

        def fail_completion_serialization(value):
            if (isinstance(value, dict)
                    and value.get("version") == subject.VERSION
                    and value.get("status") == "burned_failed"):
                raise OSError("secondary completion-serialization failure")
            return original_bytes(value)

        monkeypatch.setattr(subject, "_bytes", fail_completion_serialization)
    else:
        original_write = subject._write_phase_exclusive

        def fail_completion_publication(path, raw, **kwargs):
            if Path(path).name == "attempt_completed.json":
                raise OSError("secondary completion-publication failure")
            return original_write(path, raw, **kwargs)

        monkeypatch.setattr(
            subject, "_write_phase_exclusive", fail_completion_publication)

    expected = (
        "final_once_completion_publication_failed"
        if secondary_point == "completion_publication"
        else "secondary"
    )
    with pytest.raises(BaseException, match=expected) as raised:
        subject.run_final_once(**_run_args(
            files, development, rows, retired,
            tmp_path / ("secondary-" + secondary_point)))
    _assert_private_material_absent(raised.value, private_fragments)
    registry = tmp_path / "external" / "ledger" / subject.campaign_key()
    assert not (registry / "attempt_completed.json").exists()
    with pytest.raises(ValueError, match="already_reserved_no_retry"):
        subject._claim(identity, output=tmp_path / "secondary-retry")


def test_failed_completion_never_hashes_partial_private_output(
        tmp_path, monkeypatch):
    setup = _fake_run_setup(
        tmp_path, monkeypatch, RuntimeError("private failure after holdout"))
    files, development, rows, retired, identity, _ = setup
    monkeypatch.setattr(
        subject, "_artifact_hashes",
        lambda _output: (_ for _ in ()).throw(
            AssertionError("failed completion must not hash private output")))

    result = subject.run_final_once(**_run_args(
        files, development, rows, retired,
        tmp_path / "failed-no-output-rehash"))
    assert result["status"] == "burned_failed"
    completion = subject.read_completion(
        result["registry"],
        expected_completion_sha256=result["completion_sha256"],
        expected_identity=identity,
    )
    assert completion["artifacts"] == {}


def test_success_rejects_holdout_bytes_replaced_after_builder_attestation(
        tmp_path, monkeypatch):
    setup = _fake_run_setup(tmp_path, monkeypatch, None)
    files, development, rows, retired, _, calls = setup
    original_build = subject.holdout_api.build

    def replace_after_build(**kwargs):
        report = original_build(**kwargs)
        output = Path(kwargs["output"])
        (output / "holdout.json").write_text(
            subject.canonical({"version": "forged-after-program-known"}) + "\n",
            encoding="utf-8")
        registry = Path(subject._ledger_root()) / subject.campaign_key()
        marker = subject._read_json(
            registry / "holdout_completed.json", "holdout completion")
        marker["holdout_file_sha256"] = subject.file_hash(
            output / "holdout.json")
        (registry / "holdout_completed.json").write_text(
            subject.canonical(marker) + "\n", encoding="utf-8")
        return report

    monkeypatch.setattr(subject.holdout_api, "build", replace_after_build)
    result = subject.run_final_once(**_run_args(
        files, development, rows, retired, tmp_path / "holdout-swap"))
    assert result["status"] == "burned_failed"
    assert "audit" not in calls


def test_success_rejects_audit_output_changed_after_memory_attestation(
        tmp_path, monkeypatch):
    setup = _fake_run_setup(tmp_path, monkeypatch, None)
    files, development, rows, retired, _, _ = setup
    original_attestation = subject._audit_output_attestation

    def replace_after_attestation(report):
        attestation = original_attestation(report)
        output = tmp_path / "audit-swap" / "explanation_audit" / "report.json"
        output.write_text(
            subject.canonical({"status": "passed", "forged": True}) + "\n",
            encoding="utf-8")
        return attestation

    monkeypatch.setattr(
        subject, "_audit_output_attestation", replace_after_attestation)
    result = subject.run_final_once(**_run_args(
        files, development, rows, retired, tmp_path / "audit-swap"))
    assert result["status"] == "burned_failed"


def test_phase_marker_extra_private_field_cannot_enter_passing_completion(
        tmp_path, monkeypatch):
    setup = _fake_run_setup(tmp_path, monkeypatch, None)
    files, development, rows, retired, _, _ = setup
    original_audit = subject.audit_api.audit

    def add_private_field(**kwargs):
        report = original_audit(**kwargs)
        registry = Path(subject._ledger_root()) / subject.campaign_key()
        marker_path = registry / "audit_completed.json"
        marker = subject._read_json(marker_path, "audit completion")
        marker["selection_salt"] = "private-salt-must-not-be-accepted"
        marker_path.write_text(
            subject.canonical(marker) + "\n", encoding="utf-8")
        return report

    monkeypatch.setattr(subject.audit_api, "audit", add_private_field)
    result = subject.run_final_once(**_run_args(
        files, development, rows, retired, tmp_path / "phase-extra"))
    assert result["status"] == "burned_failed"


def test_read_completion_rejects_post_success_output_tamper(
        tmp_path, monkeypatch):
    setup = _fake_run_setup(tmp_path, monkeypatch, None)
    files, development, rows, retired, identity, _ = setup
    monkeypatch.setattr(
        subject.audit_api, "read_saved_report",
        lambda *args, **kwargs: {"status": "passed"})
    monkeypatch.setattr(
        subject.audit_api, "replay_saved_audit",
        lambda *args, **kwargs: {"status": "passed"})
    output = tmp_path / "post-success-tamper"
    result = subject.run_final_once(**_run_args(
        files, development, rows, retired, output))
    assert result["status"] == "completed_passed"
    (output / "fresh_holdout" / "holdout.json").write_text(
        subject.canonical({"private-fingerprint": "tampered"}) + "\n",
        encoding="utf-8")
    with pytest.raises(
            subject._FinalCompletionPrivatePhaseError,
            match="^final_completion_private_phase_failed$"):
        subject.read_completion(
            result["registry"],
            expected_completion_sha256=result["completion_sha256"],
            expected_identity=identity)


def test_read_completion_rejects_extra_registry_field(tmp_path, monkeypatch):
    setup = _fake_run_setup(tmp_path, monkeypatch, None)
    files, development, rows, retired, identity, _ = setup
    monkeypatch.setattr(
        subject.audit_api, "read_saved_report",
        lambda *args, **kwargs: {"status": "passed"})
    monkeypatch.setattr(
        subject.audit_api, "replay_saved_audit",
        lambda *args, **kwargs: {"status": "passed"})
    result = subject.run_final_once(**_run_args(
        files, development, rows, retired, tmp_path / "completion-extra"))
    registry = Path(result["registry"])
    completion_path = registry / "attempt_completed.json"
    completion = subject._read_json(completion_path, "completion")
    completion["selection_salt"] = "must-be-rejected"
    completion_path.write_text(
        subject.canonical(completion) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="completion schema differs"):
        subject.read_completion(
            registry, expected_completion_sha256=subject.file_hash(completion_path),
            expected_identity=identity)


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


def test_successful_completion_rechecks_sources_after_staging_before_publish(
        tmp_path, monkeypatch):
    setup = _fake_run_setup(tmp_path, monkeypatch, None)
    files, development, rows, retired, _, calls = setup
    monkeypatch.setattr(
        subject.audit_api, "read_saved_report",
        lambda *args, **kwargs: calls.append("audit_strict_read") or {
            "status": "passed"})
    monkeypatch.setattr(
        subject.audit_api, "replay_saved_audit",
        lambda *args, **kwargs: calls.append("physical_replay") or {
            "status": "passed"})
    changed = {"value": False}
    monkeypatch.setattr(
        subject, "producer_sources",
        lambda: {"source.py": ("b" if changed["value"] else "a") * 64})
    original = subject._write_phase_exclusive

    def drift_after_staging(path, raw, **kwargs):
        if Path(path).name == "attempt_completed.json":
            changed["value"] = True
        return original(path, raw, **kwargs)

    monkeypatch.setattr(subject, "_write_phase_exclusive", drift_after_staging)
    with pytest.raises(RuntimeError, match="source closure changed"):
        subject.run_final_once(**_run_args(
            files, development, rows, retired,
            tmp_path / "completion-source-drift"))
    registry = tmp_path / "external" / "ledger" / subject.campaign_key()
    assert not (registry / "attempt_completed.json").exists()


def test_candidate_replacement_after_strict_refit_burns_attempt(
        tmp_path, monkeypatch):
    setup = _fake_run_setup(tmp_path, monkeypatch, None)
    files, development, rows, retired, identity, calls = setup
    original_audit = subject.audit_api.audit

    def replacing_audit(**kwargs):
        files["program"].write_bytes(b"different self-consistent candidate")
        return original_audit(**kwargs)

    monkeypatch.setattr(subject.audit_api, "audit", replacing_audit)
    monkeypatch.setattr(
        subject.audit_api, "read_saved_report",
        lambda *args, **kwargs: {"status": "passed"})
    monkeypatch.setattr(
        subject.audit_api, "replay_saved_audit",
        lambda *args, **kwargs: {"status": "passed"})
    result = subject.run_final_once(**_run_args(
        files, development, rows, retired, tmp_path / "candidate-replaced"))
    assert result["status"] == "burned_failed"
    assert result["reason"] == "historical_final_private_phase_failed"
    with pytest.raises(ValueError, match="already_reserved_no_retry"):
        subject._claim(identity, output=tmp_path / "retry")


def test_expansion_collection_report_replacement_after_authentication_burns_attempt(
        tmp_path, monkeypatch):
    setup = _fake_run_setup(tmp_path, monkeypatch, None)
    files, development, rows, retired, identity, calls = setup
    original_audit = subject.audit_api.audit

    def replacing_audit(**kwargs):
        files["candidate_expansion_source_collection_report_json"].write_bytes(
            b"different source expansion collection report")
        return original_audit(**kwargs)

    monkeypatch.setattr(subject.audit_api, "audit", replacing_audit)
    monkeypatch.setattr(
        subject.audit_api, "read_saved_report",
        lambda *args, **kwargs: {"status": "passed"})
    monkeypatch.setattr(
        subject.audit_api, "replay_saved_audit",
        lambda *args, **kwargs: {"status": "passed"})
    result = subject.run_final_once(**_run_args(
        files, development, rows, retired,
        tmp_path / "expansion-collection-report-replaced"))
    assert result["status"] == "burned_failed"
    assert result["reason"] == "historical_final_private_phase_failed"
    with pytest.raises(ValueError, match="already_reserved_no_retry"):
        subject._claim(identity, output=tmp_path / "retry")


def test_fixed_input_toctou_before_candidate_publish_keeps_completion_readable(
        tmp_path, monkeypatch):
    setup = _fake_run_setup(tmp_path, monkeypatch, None)
    files, development, rows, retired, identity, calls = setup
    original_authenticate = subject._authenticate_candidate_after_claim

    def replace_actor_after_strict_refit(**kwargs):
        result = original_authenticate(**kwargs)
        files["actor"].write_bytes(b"post-refit fixed-input replacement")
        return result

    monkeypatch.setattr(
        subject, "_authenticate_candidate_after_claim",
        replace_actor_after_strict_refit)
    result = subject.run_final_once(**_run_args(
        files, development, rows, retired, tmp_path / "fixed-input-toctou"))
    assert result["status"] == "burned_failed"
    assert result["reason"] == "historical_final_private_phase_failed"
    assert calls == ["rcpd_strict_refit"]
    registry = Path(result["registry"])
    assert not (registry / "candidate_authenticated.json").exists()
    completion = subject.read_completion(
        registry, expected_completion_sha256=result["completion_sha256"],
        expected_identity=identity)
    assert completion["candidate_authentication_status"] is None
    assert completion["candidate_authenticated_sha256"] is None
    assert completion["candidate_artifacts"] is None
    assert completion["phase_receipts"] == {}


def test_source_closure_drift_after_fixed_claim_burns_without_output(
        tmp_path, monkeypatch):
    setup = _fake_run_setup(tmp_path, monkeypatch, None)
    files, development, rows, retired, _, calls = setup
    frozen = {"source.py": "a" * 64}
    changed = {"source.py": "b" * 64}
    source_calls = 0

    def drifting_sources():
        nonlocal source_calls
        source_calls += 1
        return frozen if source_calls == 1 else changed

    monkeypatch.setattr(subject, "producer_sources", drifting_sources)
    output = tmp_path / "preclaim-source-drift"
    result = subject.run_final_once(**_run_args(
        files, development, rows, retired, output))
    assert result["status"] == "burned_failed"
    assert result["reason"] == "historical_final_private_phase_failed"
    assert (tmp_path / "permanent" / "campaign.anchor").is_file()
    assert (Path(result["registry"]) / "attempt_completed.json").is_file()
    assert calls == []


def test_wrong_projection_is_rejected_after_irrevocable_fixed_claim(
        tmp_path, monkeypatch):
    actual_preclaim = subject._preclaim_identity
    setup = _fake_run_setup(tmp_path, monkeypatch, None)
    files, development, rows, projection_paths, _, calls = setup
    monkeypatch.setattr(subject, "_preclaim_identity", actual_preclaim)
    monkeypatch.setattr(
        subject, "EXPECTED_RETIRED_IDENTITY_PROJECTION_SHA256", "f" * 64)
    output = tmp_path / "wrong-projection"
    result = subject.run_final_once(**_run_args(
        files, development, rows, projection_paths, output))
    assert result["status"] == "burned_failed"
    assert result["reason"] == "historical_final_private_phase_failed"
    assert (tmp_path / "permanent" / "campaign.anchor").is_file()
    assert (Path(result["registry"]) / "attempt_completed.json").is_file()
    assert calls == []


def test_source_closure_drift_after_refit_burns_before_candidate_publication(
        tmp_path, monkeypatch):
    setup = _fake_run_setup(tmp_path, monkeypatch, None)
    files, development, rows, retired, _, calls = setup
    frozen = {"source.py": "a" * 64}
    changed = {"source.py": "b" * 64}
    drifted = False
    authenticate = subject._authenticate_candidate_after_claim

    monkeypatch.setattr(
        subject, "producer_sources", lambda: changed if drifted else frozen)

    def drift_after_authentication(**kwargs):
        nonlocal drifted
        result = authenticate(**kwargs)
        drifted = True
        return result

    monkeypatch.setattr(
        subject, "_authenticate_candidate_after_claim",
        drift_after_authentication)
    result = subject.run_final_once(**_run_args(
        files, development, rows, retired, tmp_path / "post-refit-source-drift"))
    assert result["status"] == "burned_failed"
    assert result["reason"] == "historical_final_private_phase_failed"
    assert calls == ["rcpd_strict_refit"]
    registry = Path(result["registry"])
    assert not (registry / "candidate_authenticated.json").exists()
    completion = subject.read_completion(
        registry, expected_completion_sha256=result["completion_sha256"],
        expected_identity=subject._campaign_identity())
    assert completion["producer_sources"] == frozen
    assert completion["candidate_authenticated_sha256"] is None


def test_implicit_strict_reader_input_drift_burns_before_candidate_publication(
        tmp_path, monkeypatch):
    setup = _fake_run_setup(tmp_path, monkeypatch, None)
    files, development, rows, retired, _, calls = setup
    authenticate = subject._authenticate_candidate_after_claim

    def drift_manifest_validation_after_authentication(**kwargs):
        result = authenticate(**kwargs)
        (tmp_path / "implicit-validation.json").write_bytes(
            b"replaced implicit validation")
        return result

    monkeypatch.setattr(
        subject, "_authenticate_candidate_after_claim",
        drift_manifest_validation_after_authentication)
    result = subject.run_final_once(**_run_args(
        files, development, rows, retired, tmp_path / "implicit-input-drift"))
    assert result["status"] == "burned_failed"
    assert result["reason"] == "historical_final_private_phase_failed"
    assert calls == ["rcpd_strict_refit"]
    registry = Path(result["registry"])
    assert not (registry / "candidate_authenticated.json").exists()


def test_candidate_marker_partial_payload_failure_is_readable_burned_ledger(
        tmp_path, monkeypatch):
    setup = _fake_run_setup(tmp_path, monkeypatch, None)
    files, development, rows, retired, identity, calls = setup
    original_payload = subject._write_phase_payload
    payload_calls = 0

    def fail_during_candidate_payload(stream, raw):
        nonlocal payload_calls
        payload_calls += 1
        if payload_calls == 1:
            stream.write(raw[: max(1, len(raw) // 2)])
            stream.flush()
            raise OSError("synthetic candidate marker payload failure")
        return original_payload(stream, raw)

    monkeypatch.setattr(subject, "_write_phase_payload", fail_during_candidate_payload)
    result = subject.run_final_once(**_run_args(
        files, development, rows, retired, tmp_path / "marker-io-failure"))
    assert result["status"] == "burned_failed"
    assert calls == ["rcpd_strict_refit"]
    completion = subject.read_completion(
        result["registry"],
        expected_completion_sha256=result["completion_sha256"],
        expected_identity=identity)
    assert completion["candidate_authenticated_sha256"] is None
    assert completion["candidate_artifacts"] is None
    assert completion["phase_receipts"] == {}
    assert completion["uncommitted_phase_residues"] == {}
    registry = Path(result["registry"])
    assert not (registry / "candidate_authenticated.json").exists()
    assert not (registry / ".candidate_authenticated.json.partial").exists()
    with pytest.raises(ValueError, match="already_reserved_no_retry"):
        subject._claim(identity, output=tmp_path / "retry-after-io-failure")


def test_candidate_marker_post_publish_failure_is_readable_burned_residue(
        tmp_path, monkeypatch):
    setup = _fake_run_setup(tmp_path, monkeypatch, None)
    files, development, rows, retired, identity, calls = setup
    original_write = subject._write_phase_exclusive

    def fail_after_publish(path, raw, **kwargs):
        original_write(path, raw, **kwargs)
        if Path(path).name == "candidate_authenticated.json":
            raise OSError("synthetic candidate marker directory fsync failure")

    monkeypatch.setattr(subject, "_write_phase_exclusive", fail_after_publish)
    result = subject.run_final_once(**_run_args(
        files, development, rows, retired, tmp_path / "marker-post-publish-failure"))
    assert result["status"] == "burned_failed"
    assert calls == ["rcpd_strict_refit"]
    completion = subject.read_completion(
        result["registry"], expected_completion_sha256=result["completion_sha256"],
        expected_identity=identity)
    assert completion["candidate_authenticated_sha256"] is None
    assert completion["candidate_artifacts"] is None
    assert completion["phase_receipts"] == {}
    assert set(completion["uncommitted_phase_residues"]) == {
        "candidate_authenticated.json"
    }


def test_holdout_started_partial_payload_failure_keeps_burned_ledger_readable(
        tmp_path, monkeypatch):
    setup = _fake_run_setup(tmp_path, monkeypatch, None)
    files, development, rows, retired, identity, calls = setup
    registry_root = tmp_path / "external" / "ledger"

    def partial_then_fail(stream, raw):
        stream.write(raw[: max(1, len(raw) // 2)])
        stream.flush()
        raise OSError("synthetic holdout phase payload failure")

    def failed_holdout(**kwargs):
        registry = registry_root / subject.campaign_key()
        subject.holdout_api._claim_marker(registry, "holdout_started.json", {
            "version": subject.holdout_api.VERSION, "status": "started_no_retry",
            "campaign_key": subject.campaign_key(),
            "candidate_identity_sha256": subject.digest(identity),
            "attempt_started_sha256": subject.file_hash(
                registry / "attempt_started.json"),
            "candidate_authenticated_sha256": subject.file_hash(
                registry / "candidate_authenticated.json"),
        })

    monkeypatch.setattr(
        subject.holdout_api, "_write_phase_payload", partial_then_fail)
    monkeypatch.setattr(subject.holdout_api, "build", failed_holdout)
    result = subject.run_final_once(**_run_args(
        files, development, rows, retired, tmp_path / "holdout-phase-failure"))
    assert result["status"] == "burned_failed"
    completion = subject.read_completion(
        result["registry"], expected_completion_sha256=result["completion_sha256"],
        expected_identity=identity)
    assert set(completion["phase_receipts"]) == {"candidate_authenticated.json"}
    registry = Path(result["registry"])
    assert not (registry / "holdout_started.json").exists()
    assert not (registry / ".holdout_started.json.partial").exists()


def test_audit_started_partial_payload_failure_keeps_burned_ledger_readable(
        tmp_path, monkeypatch):
    setup = _fake_run_setup(tmp_path, monkeypatch, None)
    files, development, rows, retired, identity, calls = setup
    registry_root = tmp_path / "external" / "ledger"

    def partial_then_fail(stream, raw):
        stream.write(raw[: max(1, len(raw) // 2)])
        stream.flush()
        raise OSError("synthetic audit phase payload failure")

    def failed_audit(**kwargs):
        registry = registry_root / subject.campaign_key()
        subject.audit_api._claim_marker(registry, "audit_started.json", {
            "version": subject.audit_api.VERSION, "status": "started_no_retry",
            "campaign_key": subject.campaign_key(),
            "candidate_identity_sha256": subject.digest(identity),
            "attempt_started_sha256": subject.file_hash(
                registry / "attempt_started.json"),
            "holdout_completed_sha256": subject.file_hash(
                registry / "holdout_completed.json"),
            "candidate_authenticated_sha256": subject.file_hash(
                registry / "candidate_authenticated.json"),
            "candidate_artifacts_sha256": subject.digest(
                subject._candidate_artifact_hashes(files, rows)),
        })

    monkeypatch.setattr(subject.audit_api, "_write_phase_payload", partial_then_fail)
    monkeypatch.setattr(subject.audit_api, "audit", failed_audit)
    result = subject.run_final_once(**_run_args(
        files, development, rows, retired, tmp_path / "audit-phase-failure"))
    assert result["status"] == "burned_failed"
    completion = subject.read_completion(
        result["registry"], expected_completion_sha256=result["completion_sha256"],
        expected_identity=identity)
    assert set(completion["phase_receipts"]) == {
        "candidate_authenticated.json", "holdout_started.json",
        "historical_exclusion_started.json",
        "historical_exclusion_completed.json",
        "holdout_completed.json",
    }
    registry = Path(result["registry"])
    assert not (registry / "audit_started.json").exists()
    assert not (registry / ".audit_started.json.partial").exists()


def test_failed_audit_gate_produces_authentic_readable_burned_ledger(
        tmp_path, monkeypatch):
    setup = _fake_run_setup(tmp_path, monkeypatch, None)
    files, development, rows, retired, identity, calls = setup
    registry_root = tmp_path / "external" / "ledger"

    def failed_audit(**kwargs):
        key = subject.campaign_key()
        registry = registry_root / key
        common = {
            "version": subject.audit_api.VERSION,
            "campaign_key": key,
            "candidate_identity_sha256": subject.digest(identity),
            "attempt_started_sha256": subject.file_hash(
                registry / "attempt_started.json"),
            "holdout_completed_sha256": subject.file_hash(
                registry / "holdout_completed.json"),
            "candidate_authenticated_sha256": subject.file_hash(
                registry / "candidate_authenticated.json"),
            "candidate_artifacts_sha256": subject.digest(
                subject._candidate_artifact_hashes(files, rows)),
        }
        output = Path(kwargs["output"]); output.mkdir()
        (output / "inputs.json").write_text("{}", encoding="utf-8")
        (output / "report.json").write_text("{}", encoding="utf-8")
        (output / "evidence.npz").write_bytes(b"evidence")
        (registry / "audit_started.json").write_text(subject.canonical({
            **common, "status": "started_no_retry"}) + "\n", encoding="utf-8")
        (registry / "audit_completed.json").write_text(subject.canonical({
            **common, "status": "failed",
            "report_file_sha256": subject.file_hash(output / "report.json"),
            "evidence_file_sha256": subject.file_hash(output / "evidence.npz"),
        }) + "\n", encoding="utf-8")
        return {"status": "failed", "bindings": {"bound": True}}

    monkeypatch.setattr(subject.audit_api, "audit", failed_audit)
    monkeypatch.setattr(
        subject.audit_api, "read_saved_report",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("development gates failed")))
    result = subject.run_final_once(**_run_args(
        files, development, rows, retired, tmp_path / "failed-gate"))
    assert result["status"] == "burned_failed"
    completion = subject.read_completion(
        result["registry"],
        expected_completion_sha256=result["completion_sha256"],
        expected_identity=identity)
    assert completion["status"] == "burned_failed"
    assert completion["audit_status"] == "failed"
    assert set(completion["phase_receipts"]) == set(subject.PHASE_RECEIPT_NAMES)


def test_read_completion_binds_permanent_anchor_and_detects_tamper(
        tmp_path, monkeypatch):
    registry_root, anchor = _ledger(tmp_path, monkeypatch)
    identity = subject._campaign_identity()
    key, registry, started = subject._claim(identity, output=tmp_path / "out")
    completion = {
        "version": subject.VERSION, "key": key, "campaign_key": key,
        "candidate_identity_sha256": subject.digest(identity),
        "status": "burned_failed", "identity": identity,
        "producer_sources": subject.producer_sources(),
        "attempt_started_sha256": subject.file_hash(
            registry / "attempt_started.json"),
        "permanent_anchor_sha256": subject.file_hash(anchor),
        "phase_receipts": {}, "uncommitted_phase_residues": {},
        "completed_at": "2026-09-13T00:00:00+00:00",
        "automatic_retry": False, "retry_allowed": False,
        "reason": "synthetic failure", "output_created": False,
        "output_identity": str(tmp_path / "out"),
        "candidate_authentication_status": None,
        "candidate_authenticated_sha256": None,
        "candidate_artifacts": None, "candidate_artifacts_sha256": None,
        "holdout_status": None, "audit_status": None,
        "physical_replay_status": None, "artifacts": {}, "program_fits": 0,
        "development_authentication_refit": False, "actor_updates": 0,
        "runtime_action_override": False, "formal_ready": False,
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
