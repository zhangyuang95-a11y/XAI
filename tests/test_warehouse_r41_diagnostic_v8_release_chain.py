from __future__ import annotations

import base64
from copy import deepcopy
from hashlib import sha256
import io
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import zipfile

import pytest

from backend.training import warehouse_r41_diagnostic_admission_v6 as admission
from backend.training import warehouse_r41_diagnostic_explanation_audit_v8 as audit
from backend.training import warehouse_r41_diagnostic_fresh_final_holdout_v4 as holdout
from backend.training import warehouse_r41_diagnostic_pair_weights_v8 as weights
from backend.training import warehouse_r41_diagnostic_rcpd_v8 as rcpd
from backend.training import warehouse_r41_diagnostic_release_receipt_v6 as receipt
from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.warehouse_r41_diagnostic_boosted_tree import (
    LEAF_VALUE_SEMANTICS, MODEL_KIND, VERSION as BOOSTED_VERSION,
    R41DiagnosticBoostedTreeProgram,
)
from backend.warehouse_r41_diagnostic_public_features_v8 import (
    R41DiagnosticPublicRelationsV8,
)
from backend.warehouse_r41_diagnostic_public_tree_program_v8 import (
    GROUPS, assemble_public_tree_program_v8,
)
from ui import warehouse_alignment_r41_diagnostic_release_v8 as release
from scripts import preflight_warehouse_r41_diagnostic_render_v6 as preflight
from ui import warehouse_alignment_online_server as server


def _features():
    required = sorted(R41DiagnosticPublicRelationsV8._required_names())
    return tuple((*required, *(f"public.filler.{i}"
                  for i in range(197 - len(required)))))


def _config():
    common = {"learning_rate": 0.1, "max_iter": 20,
              "max_leaf_nodes": 15, "min_samples_leaf": 5,
              "l2_regularization": 0.1, "max_depth": 8, "max_bins": 64,
              "random_state": 17}
    return {"version": rcpd.CONFIG_VERSION, "pair_pool_multiplier": 8.0,
            "use_action_factor": True,
            "models": {name: {**common, "random_state": 17 + index}
                       for index, name in enumerate(("base", *GROUPS))},
            "mix_weights": {"narrow_passage": 0.2,
                            "shared_pickup": 0.3,
                            "shared_charger": 0.4}}


def _component(relations, name, binding, model_config):
    trees = [{"iteration": 0, "output_index": output,
              "nodes": [{"kind": "leaf", "value": float(output == 0)}]}
             for output in range(5)]
    return R41DiagnosticBoostedTreeProgram.from_dict({
        "version": BOOSTED_VERSION, "kind": MODEL_KIND,
        "feature_names": list(relations.feature_names),
        "classes": list(range(5)),
        "action_names": ["UP", "DOWN", "LEFT", "RIGHT", "WAIT"],
        "output_kind": "multiclass_logits", "baseline": [0.0] * 5,
        "leaf_value_semantics": LEAF_VALUE_SEMANTICS,
        "n_iterations": 1, "trees": trees,
        "metadata": {"diagnostic_rcpd_version": rcpd.VERSION,
                     "diagnostic_rcpd_binding_sha256": binding,
                     "component": name, "fit_rows": 20,
                     "fit_config": model_config,
                     "prediction_input": "349 deterministic public features",
                     "validation_labels_used_for_fit": False,
                     "actor_logits_used_as_program_input": False,
                     "actor_hidden_states_used": False,
                     "runtime_action_override": False}})


def _fixture(tmp_path):
    names = _features(); relations = R41DiagnosticPublicRelationsV8(names)
    config = _config(); binding = "b" * 64
    actor_sha = admission.FIXED_ACTOR_SHA256
    actor_parameters = "a" * 64
    manifest_bindings = {"manifest_file_sha256": "c" * 64,
                         "manifest_content_sha256": "d" * 64}
    components = {name: _component(relations, name, binding,
                                   config["models"][name])
                  for name in ("base", *GROUPS)}
    program = assemble_public_tree_program_v8(
        names, components["base"], components["narrow_passage"],
        components["shared_pickup"], components["shared_charger"],
        mix_weights=config["mix_weights"], routes=rcpd._fixed_routes(),
        metadata={"native_source_actor_sha256": actor_sha,
                  "source_actor_parameters_sha256": actor_parameters,
                  "source_full_manifest_bindings": manifest_bindings,
                  "diagnostic_rcpd_version": rcpd.VERSION,
                  "diagnostic_rcpd_binding_sha256": binding,
                  "fit_config_sha256": digest(config),
                  "public_feature_contract_sha256": digest(relations.contract()),
                  "pair_weight_contract_sha256": digest(weights.contract()),
                  "actions": list(admission.EXACT_ACTION_NAMES),
                  "classes": list(admission.EXACT_CLASSES),
                  "runtime_controller": "native_neural_actor_only",
                  "runtime_action_override": False,
                  "program_feedback_into_actor": False,
                  "formal_ready": False})
    path = tmp_path / "program.json"
    path.write_text(canonical(program.to_dict()) + "\n", encoding="utf-8")
    report = {"diagnostic_rcpd_binding_sha256": binding,
              "program_file_sha256": file_hash(path),
              "program_content_sha256": digest(program.to_dict()),
              "candidate": {"complexity": program.complexity(),
                            "program_content_sha256": digest(program.to_dict())}}
    return path, names, actor_sha, actor_parameters, manifest_bindings, report, config


def _validate(fixture):
    path, names, actor_sha, parameters, manifest, report, config = fixture
    return admission.validate_v8_program_identity(
        path, actor_feature_names=names, actor_sha256=actor_sha,
        actor_parameters_sha256=parameters,
        source_full_manifest_bindings=manifest,
        rcpd_report=report, fit_config=config)


def _rewrite(fixture, mutate):
    path, names, actor_sha, parameters, manifest, report, config = fixture
    payload = json.loads(path.read_text())
    mutate(payload)
    path.write_text(canonical(payload) + "\n", encoding="utf-8")
    report["program_file_sha256"] = file_hash(path)
    report["program_content_sha256"] = digest(payload)
    report["candidate"]["program_content_sha256"] = digest(payload)
    try:
        report["candidate"]["complexity"] = (
            __import__(
                "backend.warehouse_r41_diagnostic_public_tree_program_v8",
                fromlist=["R41DiagnosticPublicTreeProgramV8"])
            .R41DiagnosticPublicTreeProgramV8.from_dict(payload).complexity())
    except ValueError:
        pass


def test_external_program_identity_accepts_exact_explicit_public_tree(tmp_path):
    result = _validate(_fixture(tmp_path))
    assert result["program_file_sha256"] == file_hash(tmp_path / "program.json")
    assert result["complexity"]["component_count"] == 4
    assert len(result["public_feature_registry_sha256"]) == 64


@pytest.mark.parametrize("mutation", [
    lambda p: p["action_names"].__setitem__(0, "NORTH"),
    lambda p: p["metadata"]["actions"].__setitem__(0, "NORTH"),
    lambda p: p["metadata"]["classes"].__setitem__(0, 9),
    lambda p: p["metadata"].update(unregistered_identity="unexpected"),
    lambda p: p["base"]["metadata"].update(unregistered_identity="unexpected"),
    lambda p: p["specialists"][0]["route"].update(threshold=0.6),
    lambda p: p["specialists"][1].update(mix_weight=0.9),
    lambda p: p["metadata"].update(runtime_controller="tree_controller"),
    lambda p: p["metadata"].update(program_feedback_into_actor=True),
])
def test_external_program_identity_fails_closed_on_registry_changes(tmp_path, mutation):
    fixture = _fixture(tmp_path); _rewrite(fixture, mutation)
    with pytest.raises(ValueError):
        _validate(fixture)


def test_external_program_identity_enforces_complexity_limit(tmp_path, monkeypatch):
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(admission, "MAX_TOTAL_NODES", 1)
    with pytest.raises(ValueError, match="complexity"):
        _validate(fixture)


def test_program_must_be_json_and_rejects_pickle_marker(tmp_path):
    fixture = _fixture(tmp_path)
    path = fixture[0]
    path.write_bytes(b"\x80pickle")
    with pytest.raises(ValueError, match="Pickle"):
        _validate(fixture)


def test_bilingual_question_contract_is_strict():
    option = {"value": "WAIT", "label": {"zh": "等待", "en": "Wait"}}
    payload = {"items": [{"prompt": {"zh": "问题", "en": "Question"},
                           "options": [deepcopy(option)]} for _ in range(8)]}
    admission._validate_bilingual_question_bank(payload)
    payload["items"][3]["prompt"].pop("en")
    with pytest.raises(ValueError, match="Chinese and English"):
        admission._validate_bilingual_question_bank(payload)


def test_release_limits_and_exact_v8_source_closure(tmp_path):
    too_large_zip = tmp_path / "too-large.zip"
    too_large_zip.write_bytes(b"x" * (release.MAX_PACKAGE_BYTES + 1))
    with pytest.raises(ValueError):
        release._package_bytes(package_path=too_large_zip)
    too_large_b64 = tmp_path / "too-large.b64"
    too_large_b64.write_bytes(b"A" * (release.MAX_BASE64_BYTES + 1))
    with pytest.raises(ValueError):
        release._package_bytes(base64_path=too_large_b64)
    sources = release.release_sources()
    assert "backend/warehouse_r41_diagnostic_public_tree_program_v8.py" in sources
    assert "backend/warehouse_r41_diagnostic_boosted_tree.py" in sources
    assert "backend/training/warehouse_r41_diagnostic_admission_v6.py" in sources
    assert "backend/warehouse_r41_diagnostic_model_tree_ensemble_v2.py" not in sources
    contract = admission.package_contract()
    assert contract["maximum_package_bytes"] == 750_000
    assert contract["maximum_base64_bytes"] == 1_000_000
    assert contract["archive_compression"] == "ZIP_BZIP2"
    assert contract["archive_compresslevel"] == 9
    assert contract["pickle_allowed"] is False


def test_release_archive_uses_only_bzip2_and_rejects_deflate(monkeypatch):
    artifacts = {name: (name.encode("ascii") + b"\n") * 4
                 for name in release.ARTIFACT_PATHS}
    manifest = {"artifacts": release._artifact_records(artifacts)}
    monkeypatch.setattr(release, "_validate_manifest", lambda value: value)
    raw = release._archive_bytes(manifest, artifacts)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        assert {info.compress_type for info in archive.infolist()} == {
            zipfile.ZIP_BZIP2}
        assert all(info.create_system == 3 for info in archive.infolist())
        assert all((info.external_attr >> 16) & 0o777 == 0o600
                   for info in archive.infolist())

    tampered = dict(artifacts)
    tampered["program"] = b"X" + tampered["program"][1:]
    stream = io.BytesIO()
    manifest_raw = (release.canonical(manifest) + "\n").encode("utf-8")
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_BZIP2,
                         compresslevel=9) as archive:
        for name, item in ((release.MANIFEST_NAME, manifest_raw),
                           *((release.ARTIFACT_PATHS[key], tampered[key])
                             for key in release.ARTIFACT_PATHS)):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (0o100000 | 0o600) << 16
            archive.writestr(info, item, compress_type=zipfile.ZIP_BZIP2,
                             compresslevel=9)
    tampered_raw = stream.getvalue()
    with pytest.raises(ValueError, match="archived artifact differs"):
        release._read_archive(
            tampered_raw,
            expected_package_sha256=sha256(tampered_raw).hexdigest(),
            expected_manifest_sha256=sha256(manifest_raw).hexdigest(),
        )

    deflated = io.BytesIO()
    with zipfile.ZipFile(deflated, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(release.MANIFEST_NAME, manifest_raw)
        for name, path in release.ARTIFACT_PATHS.items():
            archive.writestr(path, artifacts[name])
    deflated_raw = deflated.getvalue()
    with pytest.raises(ValueError, match="unsafe"):
        release._read_archive(
            deflated_raw,
            expected_package_sha256=sha256(deflated_raw).hexdigest(),
            expected_manifest_sha256=sha256(manifest_raw).hexdigest(),
        )


def test_admission_rejects_credentials_participant_data_and_absolute_local_paths():
    for value in ({"api_key": "placeholder"},
                  {"participant": {"database_url": "placeholder"}},
                  {"path": str(Path("/") / "private" / "example.json")},
                  {"path": r"C:\\Users\\example\\private.json"},
                  {"token": "Bearer abcdefghijklmnop"}):
        with pytest.raises(ValueError):
            admission._reject_sensitive(value, "fixture")


def test_release_rejects_absolute_local_paths_but_accepts_public_urls():
    for value in ({"path": "/tmp/private.json"},
                  {"path": r"D:\\private\\evidence.json"}):
        with pytest.raises(ValueError, match="absolute-local"):
            release._reject_sensitive(value, "fixture")
    release._reject_sensitive(
        {"origin": "https://policylens-warehouse-study.onrender.com"},
        "fixture",
    )


def test_clean_checkout_source_closure_requires_tracked_hash_exact_files(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.invalid"],
                   cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"],
                   cwd=checkout, check=True)
    source = checkout / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "source.py"], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=checkout, check=True)
    sources = {"source.py": release.file_hash(source)}
    result = preflight.check_clean_checkout_source_closure(checkout, sources)
    assert result["clean"] is True
    assert result["tracked"] == 1

    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        preflight.check_clean_checkout_source_closure(checkout, sources)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    untracked = checkout / "untracked.py"
    untracked.write_text("VALUE = 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="untracked"):
        preflight.check_clean_checkout_source_closure(
            checkout, {**sources, "untracked.py": release.file_hash(untracked)})


def test_clean_checkout_load_uses_only_authenticated_committed_sources(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.email", "fixture@example.invalid"],
                   cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"],
                   cwd=checkout, check=True)
    module = checkout / "fixture_release.py"
    module.write_text(
        """import base64
from hashlib import sha256
from types import SimpleNamespace

class Context:
    def __init__(self):
        self.provenance = {
            "version": "fixture.v1", "release_version": "fixture-release",
            "formal_sample_eligible": False, "data_persistent": False,
        }
        self.runtime = SimpleNamespace(actor_sha256="a" * 64)
        self.release = {"formal_ready": False}
        self.scenarios = {"splits": {"play": [
            {"fingerprint": str(index) * 64, "family_id": "f" + str(index)}
            for index in range(1, 8)
        ]}}
    def close(self):
        pass

def load_online_release(*, base64_path, expected_package_sha256,
                        expected_manifest_sha256):
    raw = base64.b64decode(open(base64_path, "rb").read(), validate=True)
    assert sha256(raw).hexdigest() == expected_package_sha256
    assert len(expected_manifest_sha256) == 64
    return Context()
""",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "fixture_release.py"], cwd=checkout,
                   check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=checkout,
                   check=True)
    secret = tmp_path / "fixture.b64"
    package = b"synthetic-package"
    secret.write_bytes(base64.b64encode(package))
    secret.chmod(0o600)
    sources = {"fixture_release.py": release.file_hash(module)}
    result = preflight.check_clean_checkout_load(
        checkout, sources, base64_path=secret,
        package_sha256=sha256(package).hexdigest(),
        manifest_sha256="b" * 64, release_module="fixture_release",
    )
    assert result["version"] == "fixture.v1"
    assert result["actor_sha256"] == "a" * 64
    assert result["source_closure"]["tracked"] == 1

    dependency = checkout / "missing_dependency.py"
    dependency.write_text("VALUE = 1\n", encoding="utf-8")
    incomplete = checkout / "incomplete_release.py"
    incomplete.write_text(
        module.read_text(encoding="utf-8")
        .replace("import base64\n", "import base64\nimport missing_dependency\n"),
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "add", "missing_dependency.py", "incomplete_release.py"],
        cwd=checkout, check=True,
    )
    subprocess.run(["git", "commit", "-qm", "incomplete"], cwd=checkout,
                   check=True)
    with pytest.raises(ValueError, match="Clean-checkout"):
        preflight.check_clean_checkout_load(
            checkout,
            {"incomplete_release.py": release.file_hash(incomplete)},
            base64_path=secret, package_sha256=sha256(package).hexdigest(),
            manifest_sha256="b" * 64, release_module="incomplete_release",
        )


def test_release_schema_binds_final_once_and_v8_program_identity():
    assert admission.FIXED_ACTOR_SHA256 == release.FIXED_ACTOR_SHA256
    assert {"final_once_identity_sha256", "final_once_attempt_completed_sha256",
            "final_once_campaign_key", "final_once_permanent_anchor_sha256",
            "final_once_candidate_authenticated_sha256",
            "final_once_holdout_started_sha256",
            "final_once_holdout_completed_sha256",
            "final_once_audit_started_sha256",
            "final_once_audit_completed_sha256",
            "fresh_final_v3_exclusion_sha256",
            "fresh_final_v3_exclusion_content_sha256",
            "explanation_audit_evidence_sha256", "physical_replay_sha256",
            "program_identity_sha256", "public_feature_contract_sha256",
            "program_complexity_sha256"} <= admission.BINDING_FIELDS
    assert "retired_fresh_final_holdout_v3" not in admission.ARTIFACT_NAMES
    assert "fresh_final_v3_exclusion" in admission.ARTIFACT_NAMES
    assert {"program_action_names", "program_classes", "program_routes",
            "program_mix_weights", "program_aggregation"} <= release._IDENTITY_FIELDS
    assert release.ARCHIVE_WHITELIST == {
        "manifest.json", "artifacts/actor.npz", "artifacts/training_protocol.json",
        "artifacts/runtime_manifest.json", "artifacts/program.json",
        "artifacts/question_bank.json", "artifacts/tutorial.json"}
    assert {"public_feature_contract_sha256", "public_feature_registry_sha256",
            "program_complexity_sha256", "final_once_identity_sha256",
            "final_once_attempt_started_sha256",
            "final_once_attempt_completed_sha256",
            "final_once_campaign_key", "final_once_permanent_anchor_sha256",
            "final_once_candidate_authenticated_sha256",
            "final_once_holdout_started_sha256",
            "final_once_holdout_completed_sha256",
            "final_once_audit_started_sha256",
            "final_once_audit_completed_sha256",
            "fresh_final_v3_exclusion_sha256",
            "fresh_final_v3_exclusion_content_sha256",
            "explanation_audit_inputs_sha256",
            "explanation_audit_evidence_sha256", "explanation_audit_sha256",
            "physical_replay_sha256"} <= receipt.FIELDS


def test_final_once_ledger_artifact_paths_are_external_and_redacted(
        tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    registry = tmp_path / "ledger" / ("a" * 64)
    registry.mkdir(parents=True)
    started = registry / "attempt_started.json"
    started.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(admission, "ROOT", repo)

    path, stored = admission._artifact_path(
        started, "final-once attempt start",
        external_name="attempt_started.json",
    )
    assert path == started
    assert stored == "external/final_once_ledger/attempt_started.json"
    assert str(tmp_path) not in stored
    assert admission.FINAL_ONCE_LEDGER_ARTIFACTS == {
        "final_once_attempt_started": "attempt_started.json",
        "final_once_candidate_authenticated": "candidate_authenticated.json",
        "final_once_holdout_started": "holdout_started.json",
        "final_once_holdout_completed": "holdout_completed.json",
        "final_once_audit_started": "audit_started.json",
        "final_once_audit_completed": "audit_completed.json",
        "final_once_attempt_completed": "attempt_completed.json",
    }

    inside = repo / "attempt_started.json"
    inside.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="external final-once ledger"):
        admission._artifact_path(
            inside, "final-once attempt start",
            external_name="attempt_started.json",
        )


def test_admission_consumes_holdout_v4_four_field_v3_claim_interface(
        monkeypatch):
    monkeypatch.setattr(
        holdout.legacy_v3, "_select",
        lambda **kwargs: (
            [{"fingerprint": "e" * 64, "seed": 7}],
            [{"fingerprint": "e" * 64}],
            {"accepted": 1},
        ),
    )
    expected = admission._final_claim_binding(
        campaign_key="a" * 64,
        attempt_started_sha256="b" * 64,
        candidate_identity_sha256="c" * 64,
        candidate_authenticated_sha256="d" * 64,
    )
    tombstone = holdout._build_v3_tombstone(
        runtime=object(), actor=object(), manifest={}, selected={},
        legacy_development_hashes=set(), retired_registries=[],
        claim_binding=expected,
    )
    admission._require_v3_claim_binding(
        tombstone["claim_binding"], expected)
    assert set(tombstone["claim_binding"]) == {
        "campaign_key", "attempt_started_sha256",
        "candidate_identity_sha256", "candidate_authenticated_sha256",
    }

    missing_authentication = dict(tombstone["claim_binding"])
    missing_authentication.pop("candidate_authenticated_sha256")
    with pytest.raises(ValueError, match="claim binding"):
        admission._require_v3_claim_binding(
            missing_authentication, expected)


@pytest.mark.parametrize("changed", [
    "candidate_authenticated_sha256",
    "candidate_artifacts",
    "candidate_artifacts_sha256",
    "source_full_manifest_bindings",
    "source_full_manifest_bindings_sha256",
    "runtime_sources",
    "runtime_sources_sha256",
])
def test_admission_consumes_complete_audit_v8_candidate_and_source_interface(
        monkeypatch, changed):
    full_manifest = {"manifest.py": "a" * 64}
    runtime_sources = {"runtime.py": "b" * 64}
    candidate_artifacts = {
        filename: format(index + 3, "x") * 64
        for index, filename in enumerate(
            admission.final_once_api.CANDIDATE_ARTIFACT_KEYS)
    }
    monkeypatch.setattr(
        audit, "diagnostic_runtime_sources", lambda: dict(runtime_sources))
    runtime = SimpleNamespace(
        source_full_manifest_bindings=dict(full_manifest))

    # Construct the producer side from audit-v8's runtime-source interface,
    # independently of admission's consumer helper.
    produced = {
        "candidate_authenticated_sha256": "c" * 64,
        "candidate_artifacts": dict(sorted(candidate_artifacts.items())),
        "candidate_artifacts_sha256": audit.digest(
            dict(sorted(candidate_artifacts.items()))),
        "source_full_manifest_bindings": dict(full_manifest),
        "source_full_manifest_bindings_sha256": audit.digest(full_manifest),
        "runtime_sources": audit.diagnostic_runtime_sources(),
        "runtime_sources_sha256": audit.digest(
            audit.diagnostic_runtime_sources()),
    }
    consumed = {
        **admission._audit_candidate_bindings(
            candidate_authenticated_sha256="c" * 64,
            candidate_artifacts=candidate_artifacts,
        ),
        **admission._audit_runtime_source_bindings(runtime),
    }
    assert consumed == produced
    expected = {"actor_sha256": admission.FIXED_ACTOR_SHA256, **consumed}
    admission._require_exact_audit_bindings(dict(expected), expected)

    mismatched = deepcopy(expected)
    mismatched.pop(changed)
    with pytest.raises(ValueError, match="external bindings"):
        admission._require_exact_audit_bindings(mismatched, expected)


@pytest.mark.parametrize("producer", [holdout, audit])
def test_admission_requires_complete_recursive_producer_source_closure(
        producer):
    sources = producer.producer_sources()
    own_path = Path(producer.__file__).resolve().relative_to(admission.ROOT).as_posix()
    assert sources[own_path] == file_hash(Path(producer.__file__).resolve())
    assert len(sources) > 1
    admission._require_exact_producer_source_closure(
        dict(sources), sources, label="fixture")

    changed = dict(sources)
    first = next(iter(changed))
    changed[first] = "f" * 64 if changed[first] != "f" * 64 else "e" * 64
    with pytest.raises(ValueError, match="producer source closure"):
        admission._require_exact_producer_source_closure(
            changed, sources, label="fixture")
    missing = dict(sources)
    missing.pop(first)
    with pytest.raises(ValueError, match="producer source closure"):
        admission._require_exact_producer_source_closure(
            missing, sources, label="fixture")


def test_final_once_fixed_paths_ignore_home_and_ledger_environment(
        tmp_path, monkeypatch):
    final_api = admission.final_once_api
    expected_ledger = final_api.REGISTRY_ROOT.absolute()
    expected_anchor = final_api.PERMANENT_ANCHOR.absolute()
    monkeypatch.setenv("HOME", str(tmp_path / "substitute-home"))
    monkeypatch.setenv(final_api.LEDGER_ENV, str(tmp_path / "substitute-ledger"))
    monkeypatch.setenv(
        final_api.PERMANENT_ANCHOR_ENV, str(tmp_path / "substitute-anchor"))
    assert final_api._ledger_root() == expected_ledger
    assert final_api._anchor_path() == expected_anchor
    assert holdout._ledger_root() == expected_ledger
    assert holdout._anchor_path() == expected_anchor

    campaign_key = "a" * 64
    admission._require_fixed_final_once_registry(
        expected_ledger / campaign_key, campaign_key)
    with pytest.raises(ValueError, match="permanent registry layout"):
        admission._require_fixed_final_once_registry(
            tmp_path / "substitute-ledger" / campaign_key, campaign_key)


def test_release_receipt_reauthenticates_same_fixed_external_ledger(
        tmp_path, monkeypatch):
    final_api = admission.final_once_api
    identity = {"version": "synthetic-final-campaign.v1"}
    key = digest(identity)
    ledger_root = tmp_path / "account-ledger"
    registry = ledger_root / key
    registry.mkdir(parents=True)
    bindings = {
        "final_once_campaign_key": key,
        "final_once_identity_sha256": digest(identity),
        "final_once_permanent_anchor_sha256": "f" * 64,
    }
    artifacts = {}
    for index, (artifact_name, filename) in enumerate(
            admission.FINAL_ONCE_LEDGER_ARTIFACTS.items()):
        path = registry / filename
        path.write_text(canonical({"index": index}) + "\n", encoding="utf-8")
        value = file_hash(path)
        bindings[artifact_name + "_sha256"] = value
        artifacts[artifact_name] = {
            "path": "external/final_once_ledger/" + filename,
            "sha256": value,
        }
    monkeypatch.setattr(final_api, "_campaign_identity", lambda: dict(identity))
    monkeypatch.setattr(final_api, "campaign_key", lambda: key)
    monkeypatch.setattr(final_api, "_ledger_root", lambda: ledger_root)

    def read_completion(path, *, expected_completion_sha256,
                        expected_identity):
        assert path == registry
        assert expected_completion_sha256 == bindings[
            "final_once_attempt_completed_sha256"]
        assert expected_identity == identity
        return {"permanent_anchor_sha256": "f" * 64}

    monkeypatch.setattr(final_api, "read_completion", read_completion)
    monkeypatch.setenv("HOME", str(tmp_path / "substitute-home"))
    value = {"bindings": bindings, "artifacts": artifacts}
    receipt._validate_fixed_final_once_ledger(value)

    redirected = deepcopy(value)
    redirected["artifacts"]["final_once_attempt_started"]["path"] = (
        str(tmp_path / "substitute-home" / "attempt_started.json"))
    with pytest.raises(ValueError, match="external ledger differs"):
        receipt._validate_fixed_final_once_ledger(redirected)


def test_admission_pins_retired_v1_v2_to_expansion_file_and_content_hashes(
        tmp_path, monkeypatch):
    files = {}
    hashes = {}
    expected = {}
    for index in (1, 2):
        name = f"retired_fresh_final_holdout_v{index}"
        version = f"warehouse-r41-diagnostic-fresh-final-holdout.v{index}"
        payload = {"version": version, "status": "retired", "marker": index}
        payload["content_sha256"] = digest(payload)
        path = tmp_path / f"v{index}.json"
        path.write_text(canonical(payload) + "\n", encoding="utf-8")
        files[name] = path
        hashes[name] = file_hash(path)
        expected[version] = {
            "file_sha256": hashes[name],
            "content_sha256": payload["content_sha256"],
        }
    monkeypatch.setattr(admission, "EXPECTED_RETIRED_HOLDOUTS", expected)
    expansion = {"bindings": {"retired_holdouts": deepcopy(expected)}}
    assert admission._validate_retired_holdout_bindings(
        files, hashes, expansion) == expected

    # A replacement registry remains rejected even if an attacker also makes
    # the supplied expansion binding internally self-consistent.
    replacement = {
        "version": "warehouse-r41-diagnostic-fresh-final-holdout.v1",
        "status": "retired", "marker": "replacement",
    }
    replacement["content_sha256"] = digest(replacement)
    files["retired_fresh_final_holdout_v1"].write_text(
        canonical(replacement) + "\n", encoding="utf-8")
    replacement_hashes = dict(hashes)
    replacement_hashes["retired_fresh_final_holdout_v1"] = file_hash(
        files["retired_fresh_final_holdout_v1"])
    replacement_bindings = deepcopy(expected)
    replacement_bindings[replacement["version"]] = {
        "file_sha256": replacement_hashes["retired_fresh_final_holdout_v1"],
        "content_sha256": replacement["content_sha256"],
    }
    with pytest.raises(ValueError, match="fixed development expansion"):
        admission._validate_retired_holdout_bindings(
            files, replacement_hashes,
            {"bindings": {"retired_holdouts": replacement_bindings}},
        )


def test_admission_binds_final_candidate_to_strict_refit_marker_bytes(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    by_filename = {}
    for filename in (
            "program.json", "report.json", "rows.npz",
            "prior_v7_report.json", "prior_v7_rows.npz",
            "expansion_rows.npz", "fit_config.json"):
        path = candidate / filename
        path.write_bytes((filename + "\n").encode("ascii"))
        by_filename[filename] = path
    simple = {}
    for name in ("actor", "protocol", "manifest", "designation",
                 "selected_scenes"):
        path = tmp_path / name
        path.write_bytes((name + "\n").encode("ascii"))
        simple[name] = path
    singles = {
        **simple,
        "program": by_filename["program.json"],
        "rcpd_report": by_filename["report.json"],
        "candidate_prior_v7_report_json": by_filename["prior_v7_report.json"],
        "candidate_prior_v7_rows_npz": by_filename["prior_v7_rows.npz"],
        "candidate_expansion_rows_npz": by_filename["expansion_rows.npz"],
        "candidate_fit_config_json": by_filename["fit_config.json"],
    }
    row_paths = [by_filename["rows.npz"]]
    development_paths = []
    for index, version in enumerate((
            holdout.DEVELOPMENT_SUPPLEMENT_VERSION,
            holdout.DEVELOPMENT_EXPANSION_VERSION)):
        path = tmp_path / f"development-{index}.json"
        path.write_text(canonical({"version": version}) + "\n", encoding="utf-8")
        development_paths.append(path)
    artifacts = admission.final_once_api._candidate_artifact_hashes(
        singles, row_paths)
    campaign_key = "a" * 64
    identity_sha = "b" * 64
    start_sha = "c" * 64
    marker = {
        "version": admission.final_once_api.VERSION
            + ".candidate-authentication.v1",
        "status": "passed_strict_reader_and_refit",
        "campaign_key": campaign_key,
        "candidate_identity_sha256": identity_sha,
        "attempt_started_sha256": start_sha,
        "candidate_artifacts": artifacts,
        "candidate_artifacts_sha256": digest(artifacts),
        "rcpd_report_file_sha256": artifacts["report.json"],
        "program_file_sha256": artifacts["program.json"],
        "rows_file_sha256": artifacts["rows.npz"],
        "prior_v7_report_file_sha256": artifacts["prior_v7_report.json"],
        "prior_v7_rows_file_sha256": artifacts["prior_v7_rows.npz"],
        "expansion_rows_file_sha256": artifacts["expansion_rows.npz"],
        "fit_config_file_sha256": artifacts["fit_config.json"],
        "actor_file_sha256": file_hash(simple["actor"]),
        "protocol_file_sha256": file_hash(simple["protocol"]),
        "manifest_file_sha256": file_hash(simple["manifest"]),
        "designation_file_sha256": file_hash(simple["designation"]),
        "selected_scenes_file_sha256": file_hash(simple["selected_scenes"]),
        "development_registries": {
            json.loads(path.read_text())["version"]: file_hash(path)
            for path in development_paths
        },
        "require_passed": True, "refit": True, "formal_ready": False,
    }
    marker_path = tmp_path / "candidate_authenticated.json"
    marker_path.write_text(canonical(marker) + "\n", encoding="utf-8")
    files = {"final_once_candidate_authenticated": marker_path}
    hashes = {
        "final_once_candidate_authenticated": file_hash(marker_path),
        "final_once_attempt_started": start_sha,
        "actor": file_hash(simple["actor"]),
        "protocol": file_hash(simple["protocol"]),
        "conflict_manifest": file_hash(simple["manifest"]),
        "diagnostic_designation": file_hash(simple["designation"]),
        "selected_scenes": file_hash(simple["selected_scenes"]),
    }
    assert admission._validate_candidate_authentication_marker(
        marker, files=files, hashes=hashes, singles=singles,
        row_paths=row_paths, development_paths=development_paths,
        campaign_key=campaign_key,
        candidate_identity_sha256=identity_sha,
    ) == artifacts

    by_filename["program.json"].write_bytes(b"post-claim replacement\n")
    with pytest.raises(ValueError, match="candidate artifacts changed"):
        admission._validate_candidate_authentication_marker(
            marker, files=files, hashes=hashes, singles=singles,
            row_paths=row_paths, development_paths=development_paths,
            campaign_key=campaign_key,
            candidate_identity_sha256=identity_sha,
        )


def test_online_server_recognizes_and_dispatches_v8(monkeypatch):
    assert release.VERSION == server.R41_DIAGNOSTIC_RELEASE_CONTEXT_VERSION_V8
    assert release.VERSION in server.R41_DIAGNOSTIC_RELEASE_CONTEXT_VERSIONS
    assert (server.R41_DIAGNOSTIC_RELEASE_MODULE_V8
            == "ui.warehouse_alignment_r41_diagnostic_release_v8")
    assert server.R41_DIAGNOSTIC_RELEASE_MODULE_V8 in server.RELEASE_MODULES

    calls = []

    class Module:
        @staticmethod
        def load_online_release(**kwargs):
            calls.append(kwargs)
            return "v8-context"

    monkeypatch.setattr(server.importlib, "import_module", lambda name: (
        Module if name == server.R41_DIAGNOSTIC_RELEASE_MODULE_V8 else None))
    result = server.load_online_context(
        expected_package_sha256="a" * 64,
        expected_manifest_sha256="b" * 64,
        base64_path="private.b64",
        release_module=server.R41_DIAGNOSTIC_RELEASE_MODULE_V8,
    )
    assert result == "v8-context"
    assert calls == [{
        "expected_package_sha256": "a" * 64,
        "expected_manifest_sha256": "b" * 64,
        "package_path": None,
        "base64_path": "private.b64",
    }]


def test_online_store_classifies_v8_context_as_diagnostic(tmp_path):
    from tests.test_warehouse_r41_diagnostic_online_server import (
        diagnostic_context,
    )

    context = diagnostic_context(tmp_path)
    context.provenance["version"] = release.VERSION
    store = server.OnlineAlignmentStudyStore(
        context, database=tmp_path / "diagnostic-v8.sqlite3",
        storage_mode="ephemeral",
    )
    try:
        assert store.is_diagnostic is True
        assert store.public_release_version == release.PUBLIC_RELEASE_VERSION
    finally:
        store.close()


def test_online_server_cli_lists_v8_release_module():
    completed = subprocess.run(
        ["python", "-m", "ui.warehouse_alignment_online_server", "--help"],
        cwd=release.ROOT, check=True, capture_output=True, text=True,
    )
    assert server.R41_DIAGNOSTIC_RELEASE_MODULE_V8 in completed.stdout
