from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest

from backend.training import warehouse_r41_failure_closeout as closeout
from backend.training.warehouse_native_common import canonical, digest


ZERO = "0" * 64


def _ledger():
    boundaries = []
    previous = 0
    for step in closeout.EXPECTED_BOUNDARY_STEPS:
        boundaries.append({
            "step": step,
            "actor_sha256": "1" * 64,
            "actor_parameters_sha256": "2" * 64,
            "checkpoint_sha256": "3" * 64,
            "program_sha256": "4" * 64,
            "tree_fit_sha256": "5" * 64,
            "evaluation_sha256": "6" * 64,
            "selected": False,
            "training_segment": {
                "start_step": previous, "end_step": step,
                "sha256": "7" * 64,
            },
            "action_authority": {
                "trainable": step + 13, "equal": step + 13, "overrides": 0,
            },
        })
        previous = step
    return {
        "version": closeout.training_ledger.VERSION,
        "status": "failed_no_eligible_actor",
        "admission_eligible": False,
        "selected": None,
        "runtime_action_override": False,
        "maximum_additional_joint_steps": 2_000_000,
        "total_actual_additional_joint_steps": 2_000_000,
        "boundary_interval": 50_000,
        "source": {"actor_sha256": closeout.R3_RELEASE["actor_sha256"]},
        "boundaries": boundaries,
    }


def _fixture_build(monkeypatch, tmp_path, ledger=None):
    ledger = deepcopy(ledger or _ledger())
    monkeypatch.setattr(closeout, "ROOT", tmp_path)
    run_root = tmp_path / "run"
    run_root.mkdir()
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(canonical(ledger) + "\n", encoding="utf-8")
    ledger_sha = sha256(ledger_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        closeout, "_reconstruct_ledger",
        lambda run, path, expected: deepcopy(ledger),
    )
    monkeypatch.setattr(
        closeout, "_boundary_receipts",
        lambda run, value: [{
            "step": step, "actor_sha256": "1" * 64,
            "selected_under_frozen_gate": False,
            "action_authority": {"trainable": step, "equal": step, "overrides": 0},
            "suite_decisions": {}, "files": {},
            "evidence_tree_sha256": ZERO,
        } for step in closeout.EXPECTED_BOUNDARY_STEPS],
    )
    monkeypatch.setattr(closeout, "_diagnostic_evidence", lambda *args: {
        "partner_alias_confirmed": True,
        "protocol_source_closure_complete": False,
        "corrected_six_partner_gate": {"admission_eligible": False},
    })
    closure = {}
    for relative in closeout.OMITTED_PROTOCOL_SOURCES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
        closure[relative] = sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(closeout, "source_closure", lambda: deepcopy(closure))
    monkeypatch.setattr(closeout, "_r3_package_evidence", lambda: {
        **closeout.R3_RELEASE, "package_path": "r3.zip", "formal_ready": False,
    })
    monkeypatch.setattr(closeout, "_validate_deployment_evidence", lambda *args: {
        "r3_retention_evidence_passed": True,
        "read_only_http_methods": ["GET"], "auto_deploy": False,
    })
    return run_root, ledger_path, ledger_sha, closure


def test_terminal_failure_requires_all_40_unselected_boundaries_and_zero_override():
    value = _ledger()
    closeout._assert_terminal_failure(value)

    partial = deepcopy(value)
    partial["boundaries"].pop()
    partial["total_actual_additional_joint_steps"] = 1_950_000
    with pytest.raises(ValueError, match="exact exhausted"):
        closeout._assert_terminal_failure(partial)

    selected = deepcopy(value)
    selected["selected"] = deepcopy(selected["boundaries"][7])
    selected["admission_eligible"] = True
    selected["status"] = "selected"
    selected["boundaries"][7]["selected"] = True
    with pytest.raises(ValueError, match="exact exhausted"):
        closeout._assert_terminal_failure(selected)

    override = deepcopy(value)
    override["boundaries"][12]["action_authority"]["overrides"] = 1
    with pytest.raises(ValueError, match="incomplete or selected"):
        closeout._assert_terminal_failure(override)


def test_build_writes_one_failure_record_and_never_grants_release(monkeypatch, tmp_path):
    run, ledger_path, ledger_sha, _ = _fixture_build(monkeypatch, tmp_path)
    output = tmp_path / "closeout"
    result = closeout.build_closeout(
        run_root=run, ledger_path=ledger_path,
        expected_ledger_sha256=ledger_sha,
        diagnostic_paths={"partner_alias": "a", "energy_gate": "b", "trend": "c"},
        diagnostic_sha256={"partner_alias": ZERO, "energy_gate": ZERO, "trend": ZERO},
        output_root=output, deployment_evidence={}, allow_test_fixture=True,
    )
    assert sorted(path.name for path in output.iterdir()) == ["failure_closeout.json"]
    assert result["status"] == closeout.STATUS
    assert result["formal_ready"] is False
    assert result["internal_pilot_ready"] is False
    assert result["admission_eligible"] is False
    assert result["release_eligible"] is False
    assert result["deployment_allowed"] is False
    assert result["diagnostic_models_only"] is True
    assert result["training"]["boundary_count"] == 40
    assert result["training"]["action_override_count"] == 0
    assert all(value is False for value in
               result["prohibited_release_artifacts"].values())
    assert result["retained_online_release"][
        "render_mutation_performed_by_closeout"] is False
    assert not any((output / name).exists()
                   for name in closeout.RELEASE_ARTIFACT_NAMES)


def test_build_rejects_selected_or_partial_ledger_before_creating_output(
        monkeypatch, tmp_path):
    selected = _ledger()
    selected["status"] = "selected"
    selected["admission_eligible"] = True
    selected["selected"] = deepcopy(selected["boundaries"][0])
    selected["boundaries"][0]["selected"] = True
    run, ledger_path, ledger_sha, _ = _fixture_build(monkeypatch, tmp_path, selected)
    output = tmp_path / "closeout"
    with pytest.raises(ValueError, match="exact exhausted"):
        closeout.build_closeout(
            run_root=run, ledger_path=ledger_path,
            expected_ledger_sha256=ledger_sha,
            diagnostic_paths={}, diagnostic_sha256={}, output_root=output,
            deployment_evidence={}, allow_test_fixture=True,
        )
    assert not output.exists()


def test_saved_closeout_rejects_release_claim_and_source_drift(monkeypatch, tmp_path):
    run, ledger_path, ledger_sha, closure = _fixture_build(monkeypatch, tmp_path)
    output = tmp_path / "closeout"
    closeout.build_closeout(
        run_root=run, ledger_path=ledger_path,
        expected_ledger_sha256=ledger_sha,
        diagnostic_paths={"partner_alias": "a", "energy_gate": "b", "trend": "c"},
        diagnostic_sha256={"partner_alias": ZERO, "energy_gate": ZERO, "trend": ZERO},
        output_root=output, deployment_evidence={}, allow_test_fixture=True,
    )
    path = output / "failure_closeout.json"
    expected = sha256(path.read_bytes()).hexdigest()
    assert closeout.read_saved_closeout(path, expected_sha256=expected)["status"] \
        == closeout.STATUS

    changed = json.loads(path.read_text())
    changed["release_eligible"] = True
    path.write_text(canonical(changed) + "\n", encoding="utf-8")
    expected = sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="not fail-closed"):
        closeout.read_saved_closeout(path, expected_sha256=expected)

    changed["release_eligible"] = False
    path.write_text(canonical(changed) + "\n", encoding="utf-8")
    monkeypatch.setattr(closeout, "source_closure", lambda: {**closure, "new.py": ZERO})
    expected = sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="no longer current"):
        closeout.read_saved_closeout(path, expected_sha256=expected)


def test_diagnostics_bind_alias_energy_and_incomplete_protocol_closure(
        monkeypatch, tmp_path):
    monkeypatch.setattr(closeout, "ROOT", tmp_path)
    source = tmp_path / "backend/training/warehouse_r41_active_run.py"
    source.parent.mkdir(parents=True)
    source.write_text("runner\n", encoding="utf-8")
    source_sha = sha256(source.read_bytes()).hexdigest()
    values = {
        "partner_alias": {
            "version": "warehouse-r41-fixed-yield-alias-diagnostic.v1",
            "status": "confirmed_validation_partner_alias",
            "root_cause": {"alias_logic": "confirmed"},
            "registered_eight_group_evidence": [{"all_core_rows_identical": True}],
            "source_receipt": {
                "backend/training/warehouse_r41_active_run.py": source_sha,
            },
        },
        "energy_gate": {
            "version": "energy.v1", "status": "passed_consistency_no_attribution_bug",
            "read_only": True, "gate_subject": "robot_2_neural_actor",
            "registered_limit": 0,
            "sources": {
                "backend/training/warehouse_r41_active_run.py": source_sha,
            },
        },
        "trend": {
            "version": "warehouse-r41-training-trend-audit.v1", "read_only": True,
            "scope": {"no_frozen_source_or_training_artifact_modified": True},
            "confirmed_implementation_defects": [
                {"id": name} for name in sorted(closeout.REQUIRED_TREND_DEFECTS)
            ],
            "source_sha256": {
                "backend/training/warehouse_r41_active_run.py": source_sha,
            },
            "protocol_implementation_sources": {"some_bound_source.py": ZERO},
        },
    }
    paths = {}
    shas = {}
    for name, value in values.items():
        path = tmp_path / (name + ".json")
        path.write_text(canonical(value) + "\n", encoding="utf-8")
        paths[name] = path
        shas[name] = sha256(path.read_bytes()).hexdigest()
    result = closeout._diagnostic_evidence(paths, shas)
    assert result["partner_alias_confirmed"] is True
    assert result["protocol_source_closure_complete"] is False
    assert result["protocol_source_closure_missing"] \
        == list(closeout.OMITTED_PROTOCOL_SOURCES)
    assert result["corrected_six_partner_gate"]["admission_eligible"] is False

    values["trend"]["confirmed_implementation_defects"].pop()
    paths["trend"].write_text(canonical(values["trend"]) + "\n", encoding="utf-8")
    shas["trend"] = sha256(paths["trend"].read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="omits a registered"):
        closeout._diagnostic_evidence(paths, shas)


def test_r3_package_and_read_only_deployment_evidence_are_exact(monkeypatch, tmp_path):
    monkeypatch.setattr(closeout, "ROOT", tmp_path)
    package = tmp_path / "r3.zip"
    manifest = {
        "identities": {"actor_sha256": "a" * 64, "play_scene_count": 12},
        "formal_ready": False,
        "parent": {"version": "warehouse-alignment-local-pilot-release.v1"},
    }
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("manifest.json", canonical(manifest))
    raw_manifest = canonical(manifest).encode("utf-8")
    r3 = {
        "package_path": "r3.zip",
        "package_sha256": sha256(package.read_bytes()).hexdigest(),
        "manifest_sha256": sha256(raw_manifest).hexdigest(),
        "actor_sha256": "a" * 64, "play_scene_count": 12,
        "public_origin": "https://example.invalid",
    }
    monkeypatch.setattr(closeout, "R3_RELEASE", r3)
    checked = closeout._r3_package_evidence()
    assert checked["actor_sha256"] == "a" * 64
    render = """services:
  - name: policylens-warehouse-study
    autoDeployTrigger: \"off\"
    envVars:
      - key: WAREHOUSE_RELEASE_PACKAGE_SHA256
        value: %s
      - key: WAREHOUSE_RELEASE_MANIFEST_SHA256
        value: %s
      - key: WAREHOUSE_PUBLIC_ORIGIN
        value: https://example.invalid
""" % (r3["package_sha256"], r3["manifest_sha256"])
    evidence = {
        "origin_commit": "b" * 40,
        "origin_render_text": render,
        "origin_render_sha256": sha256(render.encode()).hexdigest(),
        "http_methods": ["GET"],
        "health": {"value": {
            "status": "ok", "formal_ready": False,
            "version": "warehouse-alignment-online-study-server.v1",
        }, "response_sha256": "c" * 64},
        "view": {"value": {
            "play_scene_count": 12,
            "provenance": {
                "service_version": "warehouse-alignment-online-study-server.v1"},
            "enrollment": {"formal_ready": False},
        }, "response_sha256": "d" * 64},
    }
    retained = closeout._validate_deployment_evidence(evidence, checked)
    assert retained["r3_retention_evidence_passed"] is True
    assert retained["read_only_http_methods"] == ["GET"]

    evidence["http_methods"] = ["POST"]
    with pytest.raises(ValueError, match="malformed"):
        closeout._validate_deployment_evidence(evidence, checked)


def test_source_closure_binds_omitted_runner_evaluator_and_release_guards():
    sources = closeout.source_closure()
    for relative in (
        "backend/training/warehouse_r41_failure_closeout.py",
        "backend/training/warehouse_r41_active_run.py",
        "backend/training/warehouse_r41_active_evaluation.py",
        "backend/training/warehouse_r41_training_ledger.py",
        "backend/training/warehouse_r41_corrected_partner_audit.py",
        "backend/training/warehouse_r41_production_admission.py",
        "scripts/run_warehouse_r41_postfreeze_release.py",
        "scripts/preflight_warehouse_r41_render.py",
        "scripts/build_warehouse_r41_failure_closeout.py",
    ):
        assert relative in sources
        assert len(sources[relative]) == 64


def test_postfreeze_preflight_rejects_failed_ledger_before_output(monkeypatch, tmp_path):
    from scripts import run_warehouse_r41_postfreeze_release as postfreeze

    actor = tmp_path / "actor.npz"
    protocol = tmp_path / "protocol.json"
    ledger = tmp_path / "ledger.json"
    dual = tmp_path / "dual.json"
    for path in (actor, protocol, ledger, dual):
        path.write_text("x", encoding="utf-8")
    args = SimpleNamespace(
        conflict_manifest=postfreeze.FROZEN_MANIFEST,
        conflict_validation=postfreeze.FROZEN_MANIFEST.parent / "validation.json",
        task_conflict_graph=postfreeze.FROZEN_MANIFEST.parent / "task_conflict_graph.json",
        expected_conflict_manifest_sha256=postfreeze.FROZEN_MANIFEST_SHA256,
        expected_conflict_validation_sha256=postfreeze.FROZEN_VALIDATION_SHA256,
        expected_task_conflict_graph_sha256=postfreeze.FROZEN_GRAPH_SHA256,
        actor=actor, protocol=protocol, training_ledger=ledger,
        dual_evaluation=dual, expected_training_ledger_sha256=sha256(
            ledger.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(postfreeze, "_regular", lambda path, label: Path(path).resolve())
    frozen_hashes = {
        Path(postfreeze.FROZEN_MANIFEST).resolve(): postfreeze.FROZEN_MANIFEST_SHA256,
        (postfreeze.FROZEN_MANIFEST.parent / "validation.json").resolve():
            postfreeze.FROZEN_VALIDATION_SHA256,
        (postfreeze.FROZEN_MANIFEST.parent / "task_conflict_graph.json").resolve():
            postfreeze.FROZEN_GRAPH_SHA256,
        ledger.resolve(): args.expected_training_ledger_sha256,
    }

    def fake_file_hash(path):
        resolved = Path(path).resolve()
        if resolved in frozen_hashes:
            return frozen_hashes[resolved]
        return sha256(resolved.read_bytes()).hexdigest()

    monkeypatch.setattr(postfreeze, "file_hash", fake_file_hash)
    monkeypatch.setattr(
        postfreeze.ledger_api, "read_saved_ledger",
        lambda *a, **kw: (_ for _ in ()).throw(ValueError("no eligible Actor")),
    )
    with pytest.raises(ValueError, match="no eligible Actor"):
        postfreeze._preflight(args, tmp_path / "never-created")
    assert not (tmp_path / "never-created").exists()
