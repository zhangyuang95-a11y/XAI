from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from scripts import run_warehouse_r41_postfreeze_release as orchestration


def _args(tmp_path: Path) -> Namespace:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    paths = {}
    filenames = {
        "conflict_manifest": "manifest.json",
        "conflict_validation": "validation.json",
        "task_conflict_graph": "task_conflict_graph.json",
    }
    for name in (
        "training_ledger", "actor", "protocol", "dual_evaluation",
        "conflict_manifest", "conflict_validation", "task_conflict_graph",
    ):
        filename = filenames.get(name, name + (".npz" if name == "actor" else ".json"))
        path = inputs / filename
        path.write_text("{}\n", encoding="utf-8")
        paths[name] = path
    return Namespace(
        check_contracts=False,
        training_ledger=paths["training_ledger"],
        expected_training_ledger_sha256="0" * 64,
        actor=paths["actor"], protocol=paths["protocol"],
        dual_evaluation=paths["dual_evaluation"],
        conflict_manifest=paths["conflict_manifest"],
        expected_conflict_manifest_sha256="1" * 64,
        conflict_validation=paths["conflict_validation"],
        expected_conflict_validation_sha256="2" * 64,
        task_conflict_graph=paths["task_conflict_graph"],
        expected_task_conflict_graph_sha256="3" * 64,
        workers=2, output_root=tmp_path / "release",
    )


def _components(args: Namespace, output: Path) -> dict[str, Path]:
    return {
        "actor": args.actor.resolve(), "protocol": args.protocol.resolve(),
        "training_ledger": args.training_ledger.resolve(),
        "dual_evaluation": args.dual_evaluation.resolve(),
        "corrected_six_partner_audit_report": (
            output / "corrected_six_partner_audit/report.json"),
        "conflict_manifest": args.conflict_manifest.resolve(),
        "conflict_validation": args.conflict_validation.resolve(),
        "task_conflict_graph": args.task_conflict_graph.resolve(),
        "dynamic_selection_report": output / "dynamic_selection/report.json",
        "selected_scenes": output / "dynamic_selection/selected_scenes.json",
        "final_rcpd_report": output / "final_rcpd/report.json",
        "final_rcpd_program": output / "final_rcpd/program.json",
        "explanation_audit_report": output / "explanation_audit/report.json",
        "question_bank": output / "question_bank/question_bank.json",
        "question_bank_report": output / "question_bank/report.json",
        "tutorial": output / "tutorial.json",
    }


def test_cli_help_is_reachable_without_release_inputs():
    result = subprocess.run(
        [sys.executable, str(Path(orchestration.__file__)), "--help"],
        cwd=orchestration.ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "--check-contracts" in result.stdout
    assert "--expected-training-ledger-sha256" in result.stdout


def test_preflight_rejects_external_hash_before_creating_output(monkeypatch, tmp_path):
    monkeypatch.setattr(orchestration, "ROOT", tmp_path)
    args = _args(tmp_path)
    monkeypatch.setattr(orchestration, "check_contracts", lambda: {"passed": True})
    monkeypatch.setattr(orchestration, "FROZEN_MANIFEST",
                        args.conflict_manifest.resolve())
    monkeypatch.setattr(orchestration, "FROZEN_MANIFEST_SHA256",
                        args.expected_conflict_manifest_sha256)
    monkeypatch.setattr(orchestration, "FROZEN_VALIDATION_SHA256",
                        args.expected_conflict_validation_sha256)
    monkeypatch.setattr(orchestration, "FROZEN_GRAPH_SHA256",
                        args.expected_task_conflict_graph_sha256)
    with pytest.raises(ValueError, match="External SHA-256 differs"):
        orchestration.run(args)
    assert not args.output_root.exists()


def test_repository_boundary_accepts_macos_var_alias(monkeypatch, tmp_path):
    canonical_root = tmp_path.resolve()
    prefix = Path("/private/var")
    try:
        relative = canonical_root.relative_to(prefix)
    except ValueError:
        pytest.skip("macOS /var alias is unavailable")
    alias_root = Path("/var") / relative
    monkeypatch.setattr(orchestration, "ROOT", alias_root)
    source = canonical_root / "component.json"
    source.write_text("{}\n", encoding="utf-8")
    assert orchestration._regular(source, "component") == source.resolve()
    output = alias_root / "release"
    assert orchestration._new_root(output) == (canonical_root / "release")


def test_hash_tree_rejects_links_and_empty_directories(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="empty"):
        orchestration._hash_tree(empty)
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="unlinked"):
        orchestration._hash_tree(link)


def test_failed_subprocess_records_failure(tmp_path):
    journal = tmp_path / "orchestration.jsonl"
    orchestration._write_new(journal, {"event": "created"})
    with pytest.raises(RuntimeError, match="failed: deliberate"):
        orchestration._run(
            journal, "deliberate",
            [sys.executable, "-c", "raise SystemExit(7)"],
            [tmp_path / "missing"],
        )
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert rows[-1]["event"] == "failed"
    assert rows[-1]["step"] == "deliberate"
    assert rows[-1]["returncode"] == 7


def test_pipeline_runs_registered_order_and_writes_hashed_receipt(
        monkeypatch, tmp_path):
    monkeypatch.setattr(orchestration, "ROOT", tmp_path)
    monkeypatch.setattr(orchestration, "check_contracts", lambda: {"passed": True})
    args = _args(tmp_path)
    args.conflict_manifest.write_text(
        '{"splits":{"tutorial":[{}]}}\n', encoding="utf-8")
    components = _components(args, args.output_root)
    monkeypatch.setattr(
        orchestration, "_preflight",
        lambda supplied, output: (components, {"selected": {"step": 50_000}}),
    )
    monkeypatch.setattr(
        orchestration, "validate_neutral_tutorial",
        lambda payload, tutorial_scene: {"tutorial_signature": "f" * 64},
    )
    calls = []

    def fake_run(journal, name, command, outputs):
        calls.append((name, list(command)))
        if name == "portable_package":
            package, encoded = outputs
            package.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(package, "w") as archive:
                archive.writestr("manifest.json", "{}\n")
            encoded.write_text("fixture-base64\n", encoding="utf-8")
            return {"recorded": {"file": "0" * 64}}
        for output in outputs:
            if output.suffix:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text("{}\n", encoding="utf-8")
            else:
                output.mkdir(parents=True, exist_ok=True)
                (output / "evidence.json").write_text("{}\n", encoding="utf-8")
        # The direct replay following tutorial generation needs both payloads.
        if name == "neutral_tutorial":
            components["tutorial"].write_text("{}\n", encoding="utf-8")
        return {"recorded": {"file": "0" * 64}}

    monkeypatch.setattr(orchestration, "_run", fake_run)
    result = orchestration.run(args)
    assert [name for name, _ in calls] == [
        "corrected_six_partner_boundary_audit", "dynamic_selection",
        "independent_final_rcpd", "explanation_audit",
        "question_bank", "neutral_tutorial", "production_admission",
        "portable_package",
    ]
    admission_command = calls[-2][1]
    for name in orchestration.admission.ARTIFACT_NAMES:
        assert "--" + name.replace("_", "-") in admission_command
    package_command = calls[-1][1]
    assert "--expected-production-admission-sha256" in package_command
    assert result["formal_ready"] is False
    assert result["human_explanation_effect_validated"] is False
    receipt_path = args.output_root / "release_receipt.json"
    assert receipt_path.is_file()
    receipt = json.loads(receipt_path.read_text())
    assert receipt["manifest_sha256"] == orchestration.sha256(b"{}\n").hexdigest()


def test_any_step_failure_removes_every_publication_boundary(monkeypatch, tmp_path):
    monkeypatch.setattr(orchestration, "ROOT", tmp_path)
    monkeypatch.setattr(orchestration, "check_contracts", lambda: {"passed": True})
    args = _args(tmp_path)
    components = _components(args, args.output_root)
    monkeypatch.setattr(
        orchestration, "_preflight",
        lambda supplied, output: (components, {"selected": {"step": 50_000}}),
    )

    def fail_first(journal, name, command, outputs):
        root = journal.parent
        for relative in (
            "production_admission.json", "warehouse_r41_online_release.zip",
            "warehouse_r41_online_release.b64", "release_receipt.json",
        ):
            (root / relative).write_text("must be removed", encoding="utf-8")
        raise RuntimeError("producer gate failed")

    monkeypatch.setattr(orchestration, "_run", fail_first)
    with pytest.raises(RuntimeError, match="producer gate failed"):
        orchestration.run(args)
    for relative in (
        "production_admission.json", "warehouse_r41_online_release.zip",
        "warehouse_r41_online_release.b64", "release_receipt.json",
    ):
        assert not (args.output_root / relative).exists()
    rows = [json.loads(line) for line in (
        args.output_root / "orchestration.jsonl").read_text().splitlines()]
    assert rows[-1]["event"] == "pipeline_blocked"
    assert rows[-1]["formal_ready"] is False
