#!/usr/bin/env python3
"""Run the complete fail-closed warehouse r4.1 post-freeze pipeline.

The command accepts an externally hashed selected training ledger and its
exact Actor/protocol/dual-evaluation components.  It creates a new output
directory, runs every post-freeze producer in order, records every immutable
output hash, then builds the admission and portable package.  A failed step
stops the pipeline and removes any admission/package boundary.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence
import zipfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.training import warehouse_r41_production_admission as admission
from backend.training import warehouse_r41_conflict_play_selection as selection_api
from backend.training import warehouse_r41_corrected_partner_audit as corrected_partner_api
from backend.training import warehouse_r41_explanation_audit as explanation_api
from backend.training import warehouse_r41_final_rcpd as final_rcpd_api
from backend.training import warehouse_r41_question_bank as question_api
from backend.training import warehouse_r41_training_ledger as ledger_api
from backend.training.warehouse_native_common import canonical, digest, file_hash
from ui.warehouse_alignment_r41_tutorial import validate_neutral_tutorial


FROZEN_MANIFEST = ROOT / "output/warehouse_native/r41_conflict_scenes_v1_20260911/manifest.json"
FROZEN_MANIFEST_SHA256 = "4e640e19ea37405c3613d8f6ea1ace036ecff5da50cfb75c7e3f07d825f5b98e"
FROZEN_VALIDATION_SHA256 = "114ca16caeef2ec3dda729201af328c1b67b1766bfa7f9cd9a528820892b64a4"
FROZEN_GRAPH_SHA256 = "57a0c14cfef7b6659685f85bf0503997294e1a0d7641907996219fdf6bdbf8a4"
VERSION = "warehouse-r41-postfreeze-orchestration.v1"


def check_contracts() -> dict[str, Any]:
    """Validate frozen public inputs and every downstream command contract."""
    from backend.training.warehouse_r41_conflict_scenarios import (
        validate_conflict_manifest,
    )
    manifest = _regular(FROZEN_MANIFEST, "frozen conflict manifest")
    validation = _regular(FROZEN_MANIFEST.parent / "validation.json",
                          "frozen conflict validation")
    graph = _regular(FROZEN_MANIFEST.parent / "task_conflict_graph.json",
                     "frozen task conflict graph")
    if (file_hash(manifest) != FROZEN_MANIFEST_SHA256
            or file_hash(validation) != FROZEN_VALIDATION_SHA256
            or file_hash(graph) != FROZEN_GRAPH_SHA256):
        raise ValueError("Frozen r4.1 scene publication hash differs")
    value = json.loads(manifest.read_text(encoding="utf-8"))
    replay = validate_conflict_manifest(value, replay=True)
    final_contract = final_rcpd_api.contract()
    explanation_contract = explanation_api.contract()
    corrected_contract = corrected_partner_api.PROTOCOL
    if (replay.get("passed") is not True
            or final_contract.get("ppo_joint_steps") != 0
            or final_contract.get("optimizer_updates") != 0
            or final_contract.get("program_feedback_into_actor") is not False
            or explanation_contract.get("tree_controls_runtime") is not False
            or explanation_contract.get("formal_ready") is not False
            or corrected_contract.get("ppo_joint_steps") != 0
            or corrected_contract.get("runtime_action_override") is not False
            or corrected_contract.get("all_committed_boundaries_replayed") is not True
            or corrected_contract.get("distinction_required_overall_and_per_suite")
                is not True
            or corrected_contract.get("candidate_actor_shutdown_count_max") != 0
            or corrected_contract.get("partners")
                != list(corrected_partner_api.PARTNERS)):
        raise ValueError("R4.1 post-freeze producer contract differs")
    package = admission.package_contract()
    return {
        "version": VERSION, "status": "contracts_replay_verified",
        "manifest_sha256": FROZEN_MANIFEST_SHA256,
        "manifest_content_sha256": value["content_sha256"],
        "selection_version": selection_api.VERSION,
        "corrected_six_partner_audit_version": corrected_partner_api.VERSION,
        "final_rcpd_version": final_rcpd_api.VERSION,
        "explanation_version": explanation_api.VERSION,
        "question_bank_version": question_api.VERSION,
        "training_ledger_version": ledger_api.VERSION,
        "admission_version": admission.VERSION,
        "package_release_version": package["release_version"],
        "release_sources_sha256": package["release_sources_sha256"],
        "source_closure_sha256": digest(admission.source_closure()),
        "formal_ready": False,
    }


def _regular(path: str | Path, label: str) -> Path:
    root = Path(ROOT).resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(label + " must be a canonical regular file")
    value = candidate.resolve()
    try:
        value.relative_to(root)
    except ValueError:
        raise ValueError(label + " must stay inside the repository") from None
    return value


def _new_root(path: str | Path) -> Path:
    root = Path(ROOT).resolve()
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = root / value
    value = value.resolve(strict=False)
    try:
        value.relative_to(root)
    except ValueError:
        raise ValueError("Output root must stay inside the repository") from None
    if (value.exists() or value.is_symlink() or value.resolve() != value
            or value.parent.is_symlink() or not value.parent.is_dir()
            or value.parent.resolve() != value.parent):
        raise ValueError("Output root must be a new canonical directory")
    return value


def _write_new(path: Path, value: Any) -> None:
    raw = (canonical(value) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                         | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def _append(path: Path, value: Any) -> None:
    raw = (canonical(value) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND
                         | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "ab") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def _hash_tree(path: Path) -> dict[str, str]:
    if path.is_symlink() or path.resolve() != path:
        raise ValueError("Post-freeze output must be canonical and unlinked")
    if path.is_file():
        return {path.name: file_hash(path)}
    if not path.is_dir():
        raise ValueError("Post-freeze output must be a regular file or directory")
    result = {}
    for child in sorted(path.rglob("*")):
        if child.is_symlink():
            raise ValueError("Post-freeze output tree contains a link")
        if child.is_dir():
            continue
        if not child.is_file() or child.resolve() != child:
            raise ValueError("Post-freeze output tree contains a special file")
        result[child.relative_to(path).as_posix()] = file_hash(child)
    if not result:
        raise ValueError("Post-freeze output directory is empty")
    return result


def _run(journal: Path, name: str, command: Sequence[str], outputs: Sequence[Path]):
    started = datetime.now(timezone.utc).isoformat()
    _append(journal, {"version": VERSION, "event": "started", "step": name,
                      "created_utc": started, "command": list(command)})
    process = subprocess.Popen(
        list(command), cwd=ROOT, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, bufsize=1,
    )
    lines = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line.rstrip("\n"))
        if len(lines) > 40:
            lines.pop(0)
    returncode = process.wait()
    if returncode != 0:
        _append(journal, {"version": VERSION, "event": "failed", "step": name,
            "returncode": returncode, "finished_utc": datetime.now(timezone.utc).isoformat(),
            "last_output_lines": lines})
        raise RuntimeError(f"r4.1 post-freeze step failed: {name} ({returncode})")
    missing = [str(path) for path in outputs if not path.exists() or path.is_symlink()]
    if missing:
        message = f"r4.1 post-freeze step omitted outputs: {name}: {missing}"
        _append(journal, {"version": VERSION, "event": "failed", "step": name,
            "returncode": returncode,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "error": message})
        raise RuntimeError(message)
    try:
        hashes = {path.relative_to(journal.parent).as_posix(): _hash_tree(path)
                  for path in outputs}
    except BaseException as error:
        _append(journal, {"version": VERSION, "event": "failed", "step": name,
            "returncode": returncode,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "error": str(error)})
        raise
    _append(journal, {"version": VERSION, "event": "completed", "step": name,
        "finished_utc": datetime.now(timezone.utc).isoformat(), "outputs": hashes})
    return hashes


def _components(args, output: Path) -> dict[str, Path]:
    manifest = _regular(args.conflict_manifest, "frozen conflict manifest")
    validation = _regular(args.conflict_validation, "frozen conflict validation")
    graph = _regular(args.task_conflict_graph, "frozen task conflict graph")
    return {
        "actor": _regular(args.actor, "selected Actor"),
        "protocol": _regular(args.protocol, "selected protocol"),
        "training_ledger": _regular(args.training_ledger, "selected training ledger"),
        "dual_evaluation": _regular(args.dual_evaluation, "selected dual evaluation"),
        "corrected_six_partner_audit_report": output / "corrected_six_partner_audit/report.json",
        "conflict_manifest": manifest,
        "conflict_validation": validation,
        "task_conflict_graph": graph,
        "dynamic_selection_report": output / "dynamic_selection/report.json",
        "selected_scenes": output / "dynamic_selection/selected_scenes.json",
        "final_rcpd_report": output / "final_rcpd/report.json",
        "final_rcpd_program": output / "final_rcpd/program.json",
        "explanation_audit_report": output / "explanation_audit/report.json",
        "question_bank": output / "question_bank/question_bank.json",
        "question_bank_report": output / "question_bank/report.json",
        "tutorial": output / "tutorial.json",
    }


def _preflight(args, output: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    components = _components(args, output)
    registered = (
        (components["conflict_manifest"], FROZEN_MANIFEST,
         args.expected_conflict_manifest_sha256, FROZEN_MANIFEST_SHA256,
         "conflict manifest"),
        (components["conflict_validation"], FROZEN_MANIFEST.parent / "validation.json",
         args.expected_conflict_validation_sha256, FROZEN_VALIDATION_SHA256,
         "conflict validation"),
        (components["task_conflict_graph"],
         FROZEN_MANIFEST.parent / "task_conflict_graph.json",
         args.expected_task_conflict_graph_sha256, FROZEN_GRAPH_SHA256,
         "task conflict graph"),
    )
    for path, required_path, expected, required_sha, label in registered:
        if path != required_path or expected != required_sha:
            raise ValueError("Only the registered frozen r4.1 " + label + " is allowed")
    for path, expected, label in (
        (components["conflict_manifest"], args.expected_conflict_manifest_sha256,
         "conflict manifest"),
        (components["conflict_validation"], args.expected_conflict_validation_sha256,
         "conflict validation"),
        (components["task_conflict_graph"], args.expected_task_conflict_graph_sha256,
         "task conflict graph"),
        (components["training_ledger"], args.expected_training_ledger_sha256,
         "training ledger"),
    ):
        if file_hash(path) != expected:
            raise ValueError("External SHA-256 differs for " + label)
    ledger = ledger_api.read_saved_ledger(
        components["training_ledger"],
        expected_sha256=args.expected_training_ledger_sha256,
        require_selected=True,
    )
    selected = ledger["selected"]
    if (Path(selected["actor_path"]).resolve() != components["actor"]
            or selected["actor_sha256"] != file_hash(components["actor"])
            or ledger["protocol_sha256"] != file_hash(components["protocol"])
            or Path(selected["evaluation_path"]).resolve()
                != components["dual_evaluation"]
            or selected["evaluation_sha256"] != file_hash(
                components["dual_evaluation"])):
        raise ValueError("Selected ledger component identities differ")
    return components, ledger


def run(args) -> dict[str, Any]:
    missing = [name for name in (
        "training_ledger", "expected_training_ledger_sha256", "actor",
        "protocol", "dual_evaluation", "output_root",
    ) if getattr(args, name, None) is None]
    if missing:
        raise ValueError("Missing run inputs: " + ", ".join(missing))
    output = _new_root(args.output_root)
    contracts = check_contracts()
    # Validate immutable inputs before creating any output state.
    components, ledger = _preflight(args, output)
    output.mkdir(mode=0o700)
    journal = output / "orchestration.jsonl"
    _write_new(journal, {"version": VERSION, "event": "preflight_passed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "external_inputs": {
            "training_ledger_sha256": args.expected_training_ledger_sha256,
            "actor_sha256": file_hash(components["actor"]),
            "protocol_sha256": file_hash(components["protocol"]),
            "dual_evaluation_sha256": file_hash(components["dual_evaluation"]),
            "conflict_manifest_sha256": args.expected_conflict_manifest_sha256,
            "conflict_validation_sha256": args.expected_conflict_validation_sha256,
            "task_conflict_graph_sha256": args.expected_task_conflict_graph_sha256,
        },
        "contracts": contracts,
        "selected_step": ledger["selected"]["step"],
    })
    python = sys.executable
    admission_path = output / "production_admission.json"
    package_path = output / "warehouse_r41_online_release.zip"
    base64_path = output / "warehouse_r41_online_release.b64"
    try:
        _run(journal, "corrected_six_partner_boundary_audit", [
            python, "-m", "backend.training.warehouse_r41_corrected_partner_audit",
            "--training-ledger", str(components["training_ledger"]),
            "--expected-training-ledger-sha256",
            args.expected_training_ledger_sha256,
            "--dual-evaluation", str(components["dual_evaluation"]),
            "--conflict-manifest", str(components["conflict_manifest"]),
            "--workers", str(args.workers),
            "--output", str(components["corrected_six_partner_audit_report"]),
        ], [components["corrected_six_partner_audit_report"]])
        _run(journal, "dynamic_selection", [
            python, "-m", "backend.training.warehouse_r41_conflict_play_selection",
            "--manifest", str(components["conflict_manifest"]),
            "--actor", str(components["actor"]),
            "--output", str(output / "dynamic_selection"),
            "--workers", str(args.workers),
        ], [output / "dynamic_selection"])
        _run(journal, "independent_final_rcpd", [
            python, "-m", "backend.training.warehouse_r41_final_rcpd",
            "--actor", str(components["actor"]),
            "--scenarios", str(components["conflict_manifest"]),
            "--output", str(output / "final_rcpd"),
        ], [output / "final_rcpd"])
        _run(journal, "explanation_audit", [
            python, "-m", "backend.training.warehouse_r41_explanation_audit",
            "--actor", str(components["actor"]),
            "--protocol", str(components["protocol"]),
            "--program", str(components["final_rcpd_program"]),
            "--scenarios", str(components["conflict_manifest"]),
            "--output", str(output / "explanation_audit"),
        ], [output / "explanation_audit"])
        _run(journal, "question_bank", [
            python, "-m", "backend.training.warehouse_r41_question_bank",
            "--actor", str(components["actor"]),
            "--protocol", str(components["protocol"]),
            "--conflict-manifest", str(components["conflict_manifest"]),
            "--expected-conflict-manifest-sha256",
            args.expected_conflict_manifest_sha256,
            "--output", str(output / "question_bank"),
        ], [output / "question_bank"])
        _run(journal, "neutral_tutorial", [
            python, "scripts/build_warehouse_r41_neutral_tutorial.py",
            "--conflict-manifest", str(components["conflict_manifest"]),
            "--expected-conflict-manifest-sha256",
            args.expected_conflict_manifest_sha256,
            "--output", str(components["tutorial"]),
        ], [components["tutorial"]])
        # Replay once before admission; admission replays it again with the
        # selected Actor runtime to bind the deployment physics.
        manifest = json.loads(components["conflict_manifest"].read_text())
        tutorial = json.loads(components["tutorial"].read_text())
        replay = validate_neutral_tutorial(
            tutorial, tutorial_scene=manifest["splits"]["tutorial"][0])
        _append(journal, {"version": VERSION, "event": "tutorial_reverified",
            "tutorial_sha256": file_hash(components["tutorial"]),
            "tutorial_signature": replay["tutorial_signature"]})

        admission_command = [python, "scripts/build_warehouse_r41_admission.py"]
        for name in admission.ARTIFACT_NAMES:
            admission_command.extend(["--" + name.replace("_", "-"),
                                      str(components[name])])
        admission_command.extend(["--output", str(admission_path)])
        _run(journal, "production_admission", admission_command, [admission_path])
        admission_sha = file_hash(admission_path)

        release_command = [python, "scripts/build_warehouse_r41_online_release.py"]
        release_flags = {
            "actor": components["actor"], "protocol": components["protocol"],
            "training-ledger": components["training_ledger"],
            "dual-evaluation": components["dual_evaluation"],
            "corrected-six-partner-audit-report": components[
                "corrected_six_partner_audit_report"],
            "conflict-manifest": components["conflict_manifest"],
            "conflict-validation": components["conflict_validation"],
            "task-conflict-graph": components["task_conflict_graph"],
            "dynamic-selection-report": components["dynamic_selection_report"],
            "selected-scenes": components["selected_scenes"],
            "final-rcpd-report": components["final_rcpd_report"],
            "program": components["final_rcpd_program"],
            "explanation-audit-report": components["explanation_audit_report"],
            "question-bank": components["question_bank"],
            "question-bank-report": components["question_bank_report"],
            "tutorial": components["tutorial"],
            "production-admission": admission_path,
            "expected-production-admission-sha256": admission_sha,
            "output-package": package_path, "output-base64": base64_path,
        }
        for name, value in release_flags.items():
            release_command.extend(["--" + name, str(value)])
        _run(journal, "portable_package", release_command,
             [package_path, base64_path])
        # Persist the exact manifest hash required by the Render startup
        # command.  The package builder already reloaded the archive; this
        # second, local derivation keeps the deployment receipt self-contained
        # and lets the deploy preflight reject a stale Render manifest value.
        with zipfile.ZipFile(package_path, "r") as archive:
            manifest_raw = archive.read("manifest.json")
        manifest_sha = sha256(manifest_raw).hexdigest()
        receipt = {
            "version": VERSION, "status": "completed_internal_pilot_release",
            "formal_ready": False, "human_explanation_effect_validated": False,
            "training_ledger_sha256": args.expected_training_ledger_sha256,
            "actor_sha256": file_hash(components["actor"]),
            "corrected_six_partner_audit_sha256": file_hash(
                components["corrected_six_partner_audit_report"]),
            "admission_sha256": admission_sha,
            "package_sha256": file_hash(package_path),
            "manifest_sha256": manifest_sha,
            "base64_sha256": file_hash(base64_path),
            "journal_sha256_before_receipt": file_hash(journal),
            "outputs": {name: str(path) for name, path in components.items()
                        if name not in {"actor", "protocol", "training_ledger",
                                        "dual_evaluation", "conflict_manifest",
                                        "conflict_validation", "task_conflict_graph"}},
        }
        _write_new(output / "release_receipt.json", receipt)
        _append(journal, {"version": VERSION, "event": "pipeline_completed",
            "release_receipt_sha256": file_hash(output / "release_receipt.json")})
        return receipt
    except BaseException as error:
        # These three files are the only publication boundaries.  Diagnostic
        # producer evidence remains for inspection, but can never be mistaken
        # for an admitted or deployable release.
        for path in (admission_path, package_path, base64_path,
                     output / "release_receipt.json"):
            path.unlink(missing_ok=True)
        try:
            _append(journal, {"version": VERSION, "event": "pipeline_blocked",
                "error_type": type(error).__name__, "error": str(error),
                "formal_ready": False})
        finally:
            raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--check-contracts", action="store_true")
    value.add_argument("--training-ledger", type=Path)
    value.add_argument("--expected-training-ledger-sha256")
    value.add_argument("--actor", type=Path)
    value.add_argument("--protocol", type=Path)
    value.add_argument("--dual-evaluation", type=Path)
    value.add_argument("--conflict-manifest", type=Path, default=FROZEN_MANIFEST)
    value.add_argument("--expected-conflict-manifest-sha256",
                       default=FROZEN_MANIFEST_SHA256)
    value.add_argument("--conflict-validation", type=Path,
                       default=FROZEN_MANIFEST.parent / "validation.json")
    value.add_argument("--expected-conflict-validation-sha256",
                       default=FROZEN_VALIDATION_SHA256)
    value.add_argument("--task-conflict-graph", type=Path,
                       default=FROZEN_MANIFEST.parent / "task_conflict_graph.json")
    value.add_argument("--expected-task-conflict-graph-sha256",
                       default=FROZEN_GRAPH_SHA256)
    value.add_argument("--workers", type=int, default=4, choices=range(1, 9))
    value.add_argument("--output-root", type=Path)
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    result = check_contracts() if args.check_contracts else run(args)
    print(canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
