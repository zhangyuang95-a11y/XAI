#!/usr/bin/env python3
"""Create the fail-closed warehouse r4 internal-pilot admission record."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.training import warehouse_r4_production_admission as production
from backend.training import warehouse_r4_question_bank
from backend.warehouse_alignment_online_explanation import OnlineAlignmentExplainer
from backend.warehouse_alignment_online_runtime import OnlineAlignmentRuntime
from ui import warehouse_alignment_r4_online_release as release


def _repo_file(value: str | Path) -> tuple[Path, str]:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.absolute()
    try:
        relative = path.relative_to(ROOT).as_posix()
    except ValueError:
        raise ValueError("R4 admission inputs must be inside the repository") from None
    checked = release._inside_root(relative)
    if checked != path or not checked.is_file():
        raise ValueError("R4 admission input is missing, linked, or noncanonical")
    return checked, relative


def _write_new(path: Path, value) -> None:
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if (path.resolve() != path or path.parent.resolve() != path.parent
            or path.parent.is_symlink() or ROOT not in path.parents):
        raise ValueError("R4 admission output path is unsafe")
    raw = (production.canonical(value) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    parent = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _load_components(args):
    actor, _ = _repo_file(args.actor)
    protocol_path, _ = _repo_file(args.protocol)
    program, _ = _repo_file(args.program)
    selected_path, _ = _repo_file(args.selected_scenes)
    question_path, _ = _repo_file(args.question_bank)
    scenario_path, _ = _repo_file(args.validation_scenarios)
    protocol = release._read_json(protocol_path)
    selection = release._read_json(selected_path)
    question = release._read_json(question_path)
    scenarios = release._read_json(scenario_path)
    release._play_scenes(selection)
    runtime = OnlineAlignmentRuntime(actor, protocol=protocol,
        expected_actor_sha256=release.file_hash(actor),
        expected_protocol_sha256=release.digest(protocol), allow_test_fixture=False)
    explainer = OnlineAlignmentExplainer(program,
        expected_program_sha256=release.file_hash(program), runtime=runtime,
        allow_test_fixture=False)
    warehouse_r4_question_bank.validate_payload(runtime, question)
    return {"actor": actor, "protocol_path": protocol_path, "program": program,
        "selected_path": selected_path, "question_path": question_path,
        "scenario_path": scenario_path, "selection": selection, "question": question,
        "scenarios": scenarios, "runtime": runtime, "explainer": explainer}


def build(args) -> dict:
    loaded = _load_components(args)
    actor, protocol_path, program = (loaded["actor"], loaded["protocol_path"],
                                     loaded["program"])
    selected_path, question_path, scenario_path = (loaded["selected_path"],
        loaded["question_path"], loaded["scenario_path"])
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = ROOT / output
    output = output.absolute()
    try:
        output_relative = output.relative_to(ROOT).as_posix()
    except ValueError:
        raise ValueError("R4 admission output must be inside the repository") from None
    if output.exists() or output.is_symlink() or ".." in Path(output_relative).parts:
        raise FileExistsError(output)

    selection, question, scenarios = (loaded["selection"], loaded["question"],
                                      loaded["scenarios"])
    runtime, explainer = loaded["runtime"], loaded["explainer"]

    supplied = {
        "training_budget": args.training_budget,
        "validation_scenarios": args.validation_scenarios,
        "active_policy": args.active_policy_report,
        "runtime_components": args.runtime_components_report,
        "high_conflict_scenes": args.high_conflict_scenes_report,
        "explanation_program": args.explanation_program_report,
        "questionnaire": args.questionnaire_report,
    }
    report_paths, relative_paths = {}, {}
    for name in production.REPORT_NAMES:
        if supplied[name] is None:
            raise ValueError(f"--{name.replace('_', '-')} report input is required when creating admission")
        report_paths[name], relative_paths[name] = _repo_file(supplied[name])
    report_hashes = {name: release.file_hash(path) for name, path in report_paths.items()}
    reports = {name: release._read_json(path) for name, path in report_paths.items()}
    context = production.component_context(runtime=runtime, explainer=explainer,
        selection=selection, question=question, scenarios=scenarios,
        selected_scenes_path=selected_path, scenarios_path=scenario_path)
    gates = production.validate_report_set(reports=reports, report_hashes=report_hashes,
        report_paths=report_paths, context=context, selection=selection,
        question=question, scenarios=scenarios)
    admission = {
        "version": release.ADMISSION_VERSION,
        "status": "local_pilot_technically_verified",
        "formal_ready": False,
        "human_explanation_effect_validated": False,
        "gates": gates,
        "bindings": {
            "actor_sha256": runtime.actor_sha256,
            "protocol_sha256": runtime.protocol_sha256,
            "runtime_signature": runtime.signature,
            "program_sha256": explainer.program_sha256,
            "explainer_signature": explainer.signature,
            "selected_scenes_sha256": release.digest(selection),
            "question_bank_sha256": release.digest(question),
            "question_bank_signature": question.get("source_bank_signature"),
        },
        "reports": production.report_bindings(relative_paths, report_hashes),
        "sources": release.release_sources(),
        "self_path": output_relative,
    }
    _write_new(output, admission)
    admission_sha256 = release.file_hash(output)
    # Exercise the exact release-side reader after persistence.  If this fails,
    # remove only the newly written admission so it can never be mistaken for
    # a valid build input.
    try:
        persisted = release._read_json(output)
        release._validate_admission(persisted, admission_path=output,
            admission_sha256=admission_sha256, runtime=runtime, explainer=explainer,
            selection=selection, question=question,
            selected_scenes_path=selected_path, scenarios_path=scenario_path)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return {"version": production.VERSION, "status": "admission_created_and_reloaded",
            "admission": str(output), "admission_sha256": admission_sha256,
            "gates": gates, "actor_sha256": runtime.actor_sha256,
            "program_sha256": explainer.program_sha256, "formal_ready": False,
            "human_explanation_effect_validated": False}


def verify(args) -> dict:
    loaded = _load_components(args)
    admission_path, _ = _repo_file(args.verify_existing)
    expected = args.expected_admission_sha256
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError("--expected-admission-sha256 must be an external lowercase SHA-256")
    if release.file_hash(admission_path) != expected:
        raise ValueError("Admission bytes differ from the external SHA-256")
    admission = release._read_json(admission_path)
    sources = release._validate_admission(
        admission, admission_path=admission_path, admission_sha256=expected,
        runtime=loaded["runtime"], explainer=loaded["explainer"],
        selection=loaded["selection"], question=loaded["question"],
        selected_scenes_path=loaded["selected_path"],
        scenarios_path=loaded["scenario_path"],
    )
    return {"version": production.VERSION, "status": "admission_reverified",
        "admission": str(admission_path), "admission_sha256": expected,
        "gates": admission["gates"], "sources_sha256": production.digest(sources),
        "actor_sha256": loaded["runtime"].actor_sha256,
        "program_sha256": loaded["explainer"].program_sha256,
        "formal_ready": False, "human_explanation_effect_validated": False}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--actor", type=Path, required=True)
    value.add_argument("--protocol", type=Path, required=True)
    value.add_argument("--program", type=Path, required=True)
    value.add_argument("--selected-scenes", type=Path, required=True)
    value.add_argument("--question-bank", type=Path, required=True)
    value.add_argument("--validation-scenarios", type=Path, required=True)
    value.add_argument("--training-budget", type=Path)
    value.add_argument("--active-policy-report", type=Path)
    value.add_argument("--runtime-components-report", type=Path)
    value.add_argument("--high-conflict-scenes-report", type=Path)
    value.add_argument("--explanation-program-report", type=Path)
    value.add_argument("--questionnaire-report", type=Path)
    mode = value.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify-existing", type=Path)
    value.add_argument("--expected-admission-sha256")
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.verify_existing is not None:
        if args.expected_admission_sha256 is None:
            raise ValueError("--expected-admission-sha256 is required with --verify-existing")
        result = verify(args)
    else:
        if args.expected_admission_sha256 is not None:
            raise ValueError("--expected-admission-sha256 applies only to --verify-existing")
        result = build(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
