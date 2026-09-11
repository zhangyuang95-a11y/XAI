"""Fail-closed admission for the warehouse r4.1 internal diagnostic.

This module is a separate publication boundary from the historical r4.1
production admission.  It authenticates the explicitly designated terminal
Actor, then requires the diagnostic scene, explanation, questionnaire and
tutorial evidence to pass.  The sole waiver is behavioral performance; this
record can never grant formal-study or formal-sample eligibility.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from backend.training import warehouse_r41_diagnostic_conflict_play_selection as selection_api
from backend.training import warehouse_r41_diagnostic_conflict_scenarios as scenes_api
from backend.training import warehouse_r41_diagnostic_designation as designation_api
from backend.training import warehouse_r41_diagnostic_explanation_audit as explanation_api
from backend.training import warehouse_r41_diagnostic_question_bank as question_api
from backend.training import warehouse_r41_diagnostic_rcpd as rcpd_api
from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.warehouse_r41_diagnostic_online_runtime import (
    R41DiagnosticOnlineAlignmentRuntime,
)
from backend.warehouse_r41_online_explanation import (
    R41DiagnosticOnlineAlignmentExplainer,
)
from env.warehouse_native.policy import NumPyNativeActor
from env.warehouse_native.r41_diagnostic_conflict import (
    CONFLICT_FAMILIES_SHA256,
    DIAGNOSTIC_CONTRACT_SHA256,
    diagnostic_scene_fingerprint,
    diagnostic_contract_receipt,
)
from ui import warehouse_alignment_r41_diagnostic_tutorial as tutorial_api


ROOT = Path(__file__).resolve().parents[2]
VERSION = "warehouse-r41-diagnostic-admission.v1"
STATUS = "admitted_internal_diagnostic"
NAMESPACE = "internal_diagnostic"

ARTIFACT_NAMES = (
    "diagnostic_designation", "actor", "protocol", "training_ledger",
    "dual_evaluation", "failure_closeout", "diagnostic_contract",
    "conflict_manifest", "conflict_validation", "dynamic_selection_report",
    "selected_scenes", "final_rcpd_report", "final_rcpd_program",
    "explanation_audit_report", "question_bank", "question_bank_report",
    "tutorial",
)
GATE_NAMES = (
    "terminal_actor_designation", "runtime_action_authority",
    "diagnostic_conflict_publication", "dynamic_six_family_scene_selection",
    "postfreeze_grouped_rcpd", "heldout_explanation_and_intervention_audit",
    "frozen_question_bank", "neutral_tutorial_replay",
    "source_and_portable_package_contract",
)
BINDING_FIELDS = frozenset((
    "diagnostic_designation_sha256", "actor_sha256",
    "actor_parameters_sha256", "actor_metadata_sha256",
    "protocol_file_sha256", "protocol_content_sha256",
    "training_ledger_sha256", "dual_evaluation_sha256",
    "failure_closeout_sha256", "diagnostic_contract_file_sha256",
    "diagnostic_contract_semantic_sha256", "diagnostic_contract_sha256",
    "conflict_graph_sha256", "conflict_families_sha256",
    "conflict_manifest_file_sha256", "conflict_manifest_content_sha256",
    "conflict_manifest_semantic_sha256", "conflict_validation_sha256",
    "dynamic_selection_report_sha256", "selected_scenes_file_sha256",
    "selected_scenes_semantic_sha256", "final_rcpd_report_sha256",
    "final_rcpd_binding_sha256", "program_sha256", "program_content_sha256",
    "explanation_audit_sha256", "runtime_signature",
    "runtime_manifest_signature", "explainer_signature",
    "question_bank_sha256", "question_bank_signature",
    "question_bank_report_sha256", "tutorial_sha256", "tutorial_signature",
    "tutorial_scene_id", "tutorial_scene_fingerprint",
    "tutorial_successor_state_sha256", "tutorial_snapshot_sha256",
    "runtime_audit_sha256", "release_sources_sha256",
    "package_contract_sha256",
))
TOP_LEVEL_FIELDS = frozenset((
    "version", "status", "admitted", "namespace", "pilot_class",
    "behavior_performance_gate_passed", "behavior_performance_gate_waived",
    "waiver_scope", "formal_ready", "formal_sample_eligible",
    "human_explanation_effect_validated", "data_persistent",
    "runtime_action_override", "internal_diagnostic_only", "test_fixture",
    "bindings", "artifacts", "gates", "sources", "package_contract",
    "self_path",
))
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_RELATIVE = re.compile(r"[A-Za-z0-9_.\-/]+\Z")
_SENSITIVE_KEY = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|secret(?:[_-]?key)?|password|passwd|"
    r"database[_-]?url|authorization|bearer[_-]?token)(?:$|[_-])", re.I,
)
_SENSITIVE_VALUE = re.compile(
    r"(?:\bsk-[A-Za-z0-9_-]{16,}|\bBearer\s+[A-Za-z0-9._-]{12,}|"
    r"\b(?:postgres(?:ql)?|mongodb(?:\+srv)?|mysql)://)", re.I,
)
_RCPD_SELECTED_FIELDS = (
    "profile", "metrics", "gate", "complexity", "fit_diagnostics",
    "program_content_sha256",
)


def _strict_json(path: Path, label: str) -> dict[str, Any]:
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or path.stat().st_size > 512 * 1024 * 1024):
        raise ValueError(label + " is missing, linked, noncanonical, or oversized")
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("Duplicate JSON field in " + label)
            result[key] = value
        return result
    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError("Non-finite JSON value in " + label + ": " + token)),
    )
    if not isinstance(value, dict):
        raise ValueError(label + " must be one JSON object")
    return value


def _artifact_path(value: str | Path, label: str) -> tuple[Path, str]:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.absolute()
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ValueError(label + " must be a canonical regular file")
    try:
        relative = path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        raise ValueError(label + " must stay inside the repository") from None
    if not _RELATIVE.fullmatch(relative) or ".." in Path(relative).parts:
        raise ValueError(label + " has an unsafe repository path")
    return path, relative


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise ValueError("Exact lowercase SHA-256 required for " + label)
    return value


def _reject_sensitive(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _SENSITIVE_KEY.search(str(key)):
                raise ValueError("Credential-like field in " + label)
            _reject_sensitive(child, label)
    elif isinstance(value, list):
        for child in value:
            _reject_sensitive(child, label)
    elif isinstance(value, str) and _SENSITIVE_VALUE.search(value):
        raise ValueError("Credential-like value in " + label)


def source_closure() -> dict[str, str]:
    from backend.training.warehouse_r4_production_admission import local_source_hashes
    return local_source_hashes((
        Path(__file__), ROOT / "scripts/build_warehouse_r41_diagnostic_admission.py",
        Path(designation_api.__file__), Path(scenes_api.__file__),
        Path(selection_api.__file__), Path(rcpd_api.__file__),
        Path(explanation_api.__file__), Path(question_api.__file__),
        ROOT / "backend/warehouse_r41_diagnostic_online_runtime.py",
        ROOT / "backend/warehouse_r41_online_explanation.py",
        Path(tutorial_api.__file__),
        ROOT / "ui/warehouse_alignment_r41_diagnostic_release.py",
    ))


def package_contract() -> dict[str, Any]:
    from ui import warehouse_alignment_r41_diagnostic_release as release
    return {
        "release_version": release.VERSION,
        "public_release_version": release.PUBLIC_RELEASE_VERSION,
        "release_status": release.STATUS,
        "admission_version": VERSION,
        "artifact_paths": deepcopy(release.ARTIFACT_PATHS),
        "archive_whitelist": sorted(release.ARCHIVE_WHITELIST),
        "maximum_package_bytes": release.MAX_PACKAGE_BYTES,
        "maximum_base64_bytes": release.MAX_BASE64_BYTES,
        "release_sources_sha256": digest(release.release_sources()),
        "requires_external_admission_sha256": True,
        "uses_compact_seven_scene_runtime_manifest": True,
        "source_full_manifest_remains_hash_bound": True,
        "behavior_performance_waiver_scope": ["behavior_performance"],
        "runtime_action_override": False,
        "data_persistent": False,
        "formal_ready": False,
        "formal_sample_eligible": False,
    }


def _diagnostic_runtime(actor: Path, protocol_path: Path, manifest_path: Path,
                        protocol: Mapping[str, Any], manifest: Mapping[str, Any]):
    content = deepcopy(dict(manifest))
    claimed = content.pop("content_sha256", None)
    runtime = R41DiagnosticOnlineAlignmentRuntime(
        actor,
        training_protocol_path=protocol_path,
        manifest_path=manifest_path,
        expected_actor_sha256=file_hash(actor),
        expected_training_protocol_file_sha256=file_hash(protocol_path),
        expected_training_protocol_content_sha256=digest(protocol),
        expected_manifest_file_sha256=file_hash(manifest_path),
        expected_manifest_content_sha256=claimed,
        expected_manifest_semantic_sha256=digest(manifest),
    )
    runtime.verify_binding()
    return runtime


def _validate_publication(files: Mapping[str, Path], hashes: Mapping[str, str],
                          manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    receipt = _strict_json(files["diagnostic_contract"], "diagnostic contract")
    if (canonical(receipt) != canonical(diagnostic_contract_receipt())
            or receipt.get("contract_sha256") != DIAGNOSTIC_CONTRACT_SHA256
            or receipt.get("conflict_families_sha256") != CONFLICT_FAMILIES_SHA256):
        raise ValueError("Diagnostic conflict contract receipt differs")
    # The frozen publication receipt includes an exhaustive replay of each
    # base split's workload screen against the designated Actor.  Recompute
    # that same evidence here: omitting ``replay_workloads`` produces an
    # otherwise-identical receipt whose ``workload_screen.replayed`` field is
    # false and therefore rejects an authentic publication.
    recomputed = scenes_api.validate_diagnostic_manifest(
        manifest,
        replay=True,
        workload_actor_path=files["actor"],
        replay_workloads=True,
    )
    saved = _strict_json(files["conflict_validation"], "diagnostic conflict validation")
    expected = {
        **recomputed,
        "artifacts": {
            "diagnostic_contract.json": hashes["diagnostic_contract"],
            "manifest.json": hashes["conflict_manifest"],
        },
    }
    if saved != expected or saved.get("passed") is not True:
        raise ValueError("Diagnostic scene publication differs from physical replay")
    return receipt


def _validate_rcpd(files: Mapping[str, Path], hashes: Mapping[str, str],
                   expected_bindings: Mapping[str, Any]) -> Mapping[str, Any]:
    """Authenticate the immutable grouped split and selected program.

    Newer producer versions expose ``read_saved_report``; the explicit checks
    below remain as defense in depth and make the admission compatible with
    the first frozen v1 evidence directory.
    """
    output = files["final_rcpd_report"].parent
    if hasattr(rcpd_api, "read_saved_report"):
        report = rcpd_api.read_saved_report(
            output,
            expected_report_sha256=hashes["final_rcpd_report"],
            actor_path=files["actor"], protocol_path=files["protocol"],
            manifest_path=files["conflict_manifest"],
            designation_path=files["diagnostic_designation"],
            expected_designation_sha256=hashes["diagnostic_designation"],
            program_path=files["final_rcpd_program"], require_passed=True,
        )
    else:
        report = _strict_json(files["final_rcpd_report"], "diagnostic RCPD report")
    evidence = report.get("evidence_artifacts", {})
    if not isinstance(evidence, Mapping) or set(evidence) != {
            "inputs.json", "rows.npz", "candidates.json"}:
        raise ValueError("Diagnostic RCPD evidence artifact set differs")
    for name, expected in evidence.items():
        path = output / name
        if (path.is_symlink() or not path.is_file() or path.parent != output
                or file_hash(path) != expected):
            raise ValueError("Diagnostic RCPD evidence changed: " + name)
    inputs = _strict_json(output / "inputs.json", "diagnostic RCPD inputs")
    if (inputs.get("version") != rcpd_api.VERSION
            or inputs.get("bindings") != expected_bindings
            or inputs.get("contract") != rcpd_api.contract()
            or inputs.get("sources") != rcpd_api.producer_sources()
            or inputs.get("formal_ready") is not False):
        raise ValueError("Diagnostic RCPD input binding differs")
    with np.load(output / "rows.npz", allow_pickle=False) as loaded:
        if set(loaded.files) != set(rcpd_api._FIELDS):
            raise ValueError("Diagnostic RCPD row schema differs")
        arrays = {name: loaded[name] for name in loaded.files}
    train = set(map(str, arrays["observation_hashes"][~arrays["split_validation"]]))
    validation = set(map(str, arrays["observation_hashes"][arrays["split_validation"]]))
    if train & validation or not bool(np.all(arrays["submitted_equal"])):
        raise ValueError("Diagnostic RCPD split leaks observations or overrides actions")
    candidates = _strict_json(output / "candidates.json", "diagnostic RCPD candidates")
    chosen = rcpd_api._choose(candidates.get("candidates", []))
    selected_projection = None if chosen is None else {
        key: deepcopy(chosen[key]) for key in _RCPD_SELECTED_FIELDS
    }
    program = _strict_json(files["final_rcpd_program"], "diagnostic RCPD program")
    execution = report.get("execution", {})
    if (report.get("version") != rcpd_api.VERSION
            or report.get("status") != "passed"
            or report.get("explanation_eligible") is not True
            or report.get("formal_ready") is not False
            or report.get("bindings") != expected_bindings
            or report.get("diagnostic_rcpd_binding_sha256")
                != candidates.get("binding_sha256")
            or report.get("exact_observation_overlap") != 0
            or report.get("selected") != selected_projection
            or chosen is None or chosen.get("gate", {}).get("passed") is not True
            or digest(program) != chosen.get("program_content_sha256")
            or report.get("program_file_sha256") != hashes["final_rcpd_program"]
            or execution.get("ppo_joint_steps") != 0
            or execution.get("optimizer_updates") != 0
            or execution.get("actor_changed") is not False
            or execution.get("runtime_action_overrides") != 0):
        raise ValueError("Exact passing postfreeze diagnostic RCPD required")
    return report


def _validate_question_report(report: Mapping[str, Any], payload: Mapping[str, Any],
                              manifest: Mapping[str, Any], manifest_sha: str,
                              runtime: R41DiagnosticOnlineAlignmentRuntime) -> None:
    expected_fields = {
        "version", "status", "actor_sha256", "protocol_sha256",
        "runtime_signature", "pool_manifest_sha256", "candidate_frames",
        "selected_scenarios", "checks", "source_bank_signature",
        "payload_sha256", "formal_ready",
        "diagnostic_manifest_content_sha256",
        "diagnostic_manifest_semantic_sha256", "source_full_manifest_bindings",
    }
    if (set(report) != expected_fields or report.get("version") != question_api.VERSION
            or report.get("status") != "candidate_ready"
            or report.get("actor_sha256") != runtime.actor_sha256
            or report.get("protocol_sha256") != runtime.protocol_sha256
            or report.get("runtime_signature") != runtime.signature
            or report.get("pool_manifest_sha256") != manifest_sha
            or report.get("selected_scenarios") != 8
            or type(report.get("candidate_frames")) is not int
            or report["candidate_frames"] < 8
            or report.get("checks") != payload.get("checks")
            or report.get("source_bank_signature")
                != payload.get("source_bank_signature")
            or report.get("payload_sha256") != digest(payload)
            or report.get("formal_ready") is not False
            or report.get("diagnostic_manifest_content_sha256")
                != manifest.get("content_sha256")
            or report.get("diagnostic_manifest_semantic_sha256") != digest(manifest)
            or report.get("source_full_manifest_bindings")
                != runtime.source_full_manifest_bindings):
        raise ValueError("Diagnostic frozen question-bank report differs")


def _runtime_audit(runtime: R41DiagnosticOnlineAlignmentRuntime,
                   selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    scenes = [selection["tutorial"], *selection["X"], *selection["Y"]]
    if (len(scenes) != 7 or len({row.get("id") for row in scenes}) != 7
            or len({row.get("family_id") for row in scenes[1:]}) != 6):
        raise ValueError("Diagnostic runtime requires one tutorial and six families")
    audit_rows = []
    player_actions = ("WAIT", "UP", "WAIT", "LEFT", "WAIT", "DOWN", "WAIT", "RIGHT")
    for scene in scenes:
        env = runtime.environment(deepcopy(scene))
        if diagnostic_scene_fingerprint(env) != scene.get("fingerprint"):
            raise ValueError("Diagnostic runtime restored another scene")
        initial = digest(env.snapshot())
        branch = runtime.counterfactual(env.snapshot(), ["WAIT"], steps=1)
        if digest(env.snapshot()) != initial or branch.get("steps_executed") != 1:
            raise ValueError("Diagnostic runtime counterfactual is not isolated")
        steps = overrides = 0
        while not env.done:
            before = digest(env.snapshot())
            actions, decision = runtime.decision(env)
            if (digest(env.snapshot()) != before
                    or decision.get("post_policy_overrides") != 0
                    or actions["robot_2"] != decision["policy_actions"]["robot_2"]):
                raise ValueError("Diagnostic runtime decision changed state or action")
            transition = runtime.step(env, player_actions[steps % len(player_actions)])
            steps += 1
            overrides += int(transition["decision"]["post_policy_overrides"])
            if (transition["submitted_actions"]["robot_2"]
                    != transition["policy_actions"]["robot_2"]
                    or transition["submitted_actions"]["robot_2"] != actions["robot_2"]):
                raise ValueError("Diagnostic runtime overrode the frozen Actor")
            if steps > runtime.config.horizon:
                raise ValueError("Diagnostic runtime exceeded its public horizon")
        if overrides != 0:
            raise ValueError("Diagnostic runtime action override count is nonzero")
        audit_rows.append({
            "id": scene["id"], "fingerprint": scene["fingerprint"],
            "family_id": scene.get("family_id"), "initial_state_sha256": initial,
            "steps": steps, "terminal_state_sha256": digest(env.snapshot()),
            "counterfactual_isolated": True, "post_policy_overrides": 0,
            "all_policy_actions_submitted_exactly": True,
        })
    return audit_rows


def validate_components(paths: Mapping[str, str | Path]) -> dict[str, Any]:
    if not isinstance(paths, Mapping) or set(paths) != set(ARTIFACT_NAMES):
        raise ValueError("Exact r4.1 diagnostic admission artifact set required")
    files, relatives = {}, {}
    for name in ARTIFACT_NAMES:
        files[name], relatives[name] = _artifact_path(paths[name], name)
    hashes = {name: file_hash(path) for name, path in files.items()}

    designation_components = {
        name: files[name] for name in designation_api.ARTIFACT_NAMES
    }
    designation = designation_api.read_saved_designation(
        files["diagnostic_designation"],
        expected_sha256=hashes["diagnostic_designation"],
        components=designation_components,
    )
    if (designation.get("behavior_performance_gate_passed") is not False
            or designation.get("behavior_performance_gate_waived") is not True
            or designation.get("waiver_scope") != ["behavior_performance"]
            or designation.get("formal_sample_eligible") is not False):
        raise ValueError("Diagnostic designation has another waiver scope")
    designated = designation["bindings"]
    if any(hashes[name] != designated[field] for name, field in (
            ("actor", "actor_sha256"), ("protocol", "protocol_file_sha256"),
            ("training_ledger", "training_ledger_sha256"),
            ("dual_evaluation", "dual_evaluation_sha256"),
            ("failure_closeout", "failure_closeout_sha256"))):
        raise ValueError("Diagnostic admission source differs from designation")

    protocol = _strict_json(files["protocol"], "diagnostic training protocol")
    manifest = _strict_json(files["conflict_manifest"], "diagnostic conflict manifest")
    contract_receipt = _validate_publication(files, hashes, manifest)
    runtime = _diagnostic_runtime(
        files["actor"], files["protocol"], files["conflict_manifest"],
        protocol, manifest,
    )
    dynamic = selection_api.read_saved_diagnostic_selection(
        files["dynamic_selection_report"].parent,
        expected_report_sha256=hashes["dynamic_selection_report"],
        actor_path=files["actor"], manifest_path=files["conflict_manifest"],
        selected_scenes_path=files["selected_scenes"], require_selected=True,
    )
    selection = _strict_json(files["selected_scenes"], "diagnostic selected scenes")
    if (dynamic.get("passed") is not True
            or selection.get("release_eligible") is not True
            or selection.get("six_distinct_conflict_families") is not True
            or selection.get("zero_action_overrides") is not True
            or selection.get("no_replacement_pickup_on_agent") is not True
            or selection.get("no_replacement_delivery_on_agent") is not True
            or selection.get("no_replacement_endpoint_on_agent") is not True):
        raise ValueError("Diagnostic dynamic six-family selection did not pass")

    actor = NumPyNativeActor(files["actor"])
    rcpd_bindings = {
        "designation_sha256": hashes["diagnostic_designation"],
        "actor_file_sha256": hashes["actor"],
        "actor_parameters_sha256": actor.metadata["actor_parameters_sha256"],
        "protocol_file_sha256": hashes["protocol"],
        "protocol_content_sha256": digest(protocol),
        "manifest_file_sha256": hashes["conflict_manifest"],
        "manifest_content_sha256": manifest["content_sha256"],
        "manifest_semantic_sha256": digest(manifest),
        "contract_sha256": digest(rcpd_api.contract()),
        "producer_sources_sha256": digest(rcpd_api.producer_sources()),
        "holdout_fingerprints_sha256": digest(sorted(
            row["fingerprint"] for row in manifest["splits"]["final_test"])),
    }
    final = _validate_rcpd(files, hashes, rcpd_bindings)
    explainer = R41DiagnosticOnlineAlignmentExplainer(
        files["final_rcpd_program"],
        expected_program_sha256=hashes["final_rcpd_program"], runtime=runtime,
    )
    explainer._assert_current(runtime)
    holdout_hash = digest(sorted(
        row["fingerprint"] for row in manifest["splits"]["final_test"]))
    explanation_bindings = {
        "actor_sha256": runtime.actor_sha256,
        "actor_parameters_sha256": actor.metadata["actor_parameters_sha256"],
        "protocol_file_sha256": hashes["protocol"],
        "protocol_content_sha256": runtime.protocol_sha256,
        "runtime_signature": runtime.signature,
        "runtime_manifest_signature": runtime.runtime_manifest_signature,
        "program_sha256": explainer.program_sha256,
        "explainer_signature": explainer.signature,
        "program_content_sha256": explainer.program_content_sha256,
        "rcpd_report_sha256": hashes["final_rcpd_report"],
        "manifest_file_sha256": hashes["conflict_manifest"],
        "manifest_content_sha256": manifest["content_sha256"],
        "manifest_semantic_sha256": digest(manifest),
        "source_full_manifest_bindings": runtime.source_full_manifest_bindings,
        "test_fixture": False,
        "contract_sha256": digest(explanation_api.contract()),
        "producer_sources_sha256": digest(explanation_api.producer_sources()),
        "holdout_fingerprints_sha256": holdout_hash,
    }
    explanation = explanation_api.read_saved_report(
        files["explanation_audit_report"].parent,
        expected_report_sha256=hashes["explanation_audit_report"],
        expected_bindings=explanation_bindings,
    )
    if (explanation.get("status") != "passed"
            or explanation.get("statistics", {}).get("passed") is not True
            or explanation.get("zero_nn_overrides") is not True
            or explanation.get("counterfactual_isolated") is not True
            or explanation.get("program_never_controls_action") is not True):
        raise ValueError("Diagnostic heldout explanation audit did not pass")

    question = _strict_json(files["question_bank"], "diagnostic question bank")
    question_report = question_api.read_saved_report(
        files["question_bank_report"].parent,
        expected_report_sha256=hashes["question_bank_report"],
        expected_question_bank_sha256=hashes["question_bank"],
        runtime=runtime, manifest=manifest,
        manifest_file_sha256=hashes["conflict_manifest"],
    )
    _validate_question_report(
        question_report, question, manifest, hashes["conflict_manifest"], runtime)

    tutorial_scene = selection["tutorial"]
    tutorial = _strict_json(files["tutorial"], "diagnostic neutral tutorial")
    expected_tutorial = tutorial_api.bindings(
        manifest, tutorial_scene,
        manifest_file_sha256=hashes["conflict_manifest"],
    )
    tutorial_result = tutorial_api.validate_neutral_tutorial(
        tutorial, tutorial_scene=tutorial_scene, runtime=runtime,
        expected_bindings=expected_tutorial,
    )
    tutorial_signature = digest(tutorial)
    if (tutorial_result.get("tutorial_signature") != tutorial_signature
            or tutorial.get("uses_final_actor") is not False):
        raise ValueError("Diagnostic neutral tutorial boundary differs")

    runtime_rows = _runtime_audit(runtime, selection)
    package = package_contract()
    sources = source_closure()
    for label, value in (
        ("designation", designation), ("protocol", protocol),
        ("scene manifest", manifest), ("scene selection", selection),
        ("RCPD", final), ("explanation", explanation),
        ("question bank", question), ("tutorial", tutorial),
    ):
        _reject_sensitive(value, label)
    bindings = {
        "diagnostic_designation_sha256": hashes["diagnostic_designation"],
        "actor_sha256": hashes["actor"],
        "actor_parameters_sha256": actor.metadata["actor_parameters_sha256"],
        "actor_metadata_sha256": digest(actor.metadata),
        "protocol_file_sha256": hashes["protocol"],
        "protocol_content_sha256": digest(protocol),
        "training_ledger_sha256": hashes["training_ledger"],
        "dual_evaluation_sha256": hashes["dual_evaluation"],
        "failure_closeout_sha256": hashes["failure_closeout"],
        "diagnostic_contract_file_sha256": hashes["diagnostic_contract"],
        "diagnostic_contract_semantic_sha256": digest(contract_receipt),
        "diagnostic_contract_sha256": DIAGNOSTIC_CONTRACT_SHA256,
        "conflict_graph_sha256": contract_receipt["conflict_graph_sha256"],
        "conflict_families_sha256": CONFLICT_FAMILIES_SHA256,
        "conflict_manifest_file_sha256": hashes["conflict_manifest"],
        "conflict_manifest_content_sha256": manifest["content_sha256"],
        "conflict_manifest_semantic_sha256": digest(manifest),
        "conflict_validation_sha256": hashes["conflict_validation"],
        "dynamic_selection_report_sha256": hashes["dynamic_selection_report"],
        "selected_scenes_file_sha256": hashes["selected_scenes"],
        "selected_scenes_semantic_sha256": digest(selection),
        "final_rcpd_report_sha256": hashes["final_rcpd_report"],
        "final_rcpd_binding_sha256": final["diagnostic_rcpd_binding_sha256"],
        "program_sha256": hashes["final_rcpd_program"],
        "program_content_sha256": explainer.program_content_sha256,
        "explanation_audit_sha256": hashes["explanation_audit_report"],
        "runtime_signature": runtime.signature,
        "runtime_manifest_signature": runtime.runtime_manifest_signature,
        "explainer_signature": explainer.signature,
        "question_bank_sha256": hashes["question_bank"],
        "question_bank_signature": question["source_bank_signature"],
        "question_bank_report_sha256": hashes["question_bank_report"],
        "tutorial_sha256": hashes["tutorial"],
        "tutorial_signature": tutorial_signature,
        "tutorial_scene_id": tutorial_scene["id"],
        "tutorial_scene_fingerprint": tutorial_scene["fingerprint"],
        "tutorial_successor_state_sha256": tutorial_scene["snapshot"]
            ["r41_diagnostic_conflict"]["binding_sha256"],
        "tutorial_snapshot_sha256": digest(tutorial_scene["snapshot"]),
        "runtime_audit_sha256": digest(runtime_rows),
        "release_sources_sha256": package["release_sources_sha256"],
        "package_contract_sha256": digest(package),
    }
    if set(bindings) != BINDING_FIELDS:
        raise AssertionError("Diagnostic admission binding schema differs")
    for name, value in bindings.items():
        if name != "tutorial_scene_id":
            _sha(value, name)
    return {
        "bindings": bindings,
        "artifacts": {
            name: {"path": relatives[name], "sha256": hashes[name]}
            for name in ARTIFACT_NAMES
        },
        "gates": {name: True for name in GATE_NAMES},
        "sources": sources,
        "package_contract": package,
        "runtime_audit": runtime_rows,
    }


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if (path.exists() or path.is_symlink() or path.resolve() != path
            or path.parent.is_symlink() or path.parent.resolve() != path.parent):
        raise ValueError("Diagnostic admission output path is unsafe")
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write((canonical(value) + "\n").encode("utf-8"))
        stream.flush(); os.fsync(stream.fileno())


def build_admission(paths: Mapping[str, str | Path], *, output: str | Path) -> dict[str, Any]:
    output_path = Path(output).expanduser()
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path = output_path.absolute()
    try:
        relative = output_path.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        raise ValueError("Diagnostic admission output must stay inside repository") from None
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    checked = validate_components(paths)
    value = {
        "version": VERSION, "status": STATUS, "admitted": True,
        "namespace": NAMESPACE, "pilot_class": NAMESPACE,
        "behavior_performance_gate_passed": False,
        "behavior_performance_gate_waived": True,
        "waiver_scope": ["behavior_performance"],
        "formal_ready": False, "formal_sample_eligible": False,
        "human_explanation_effect_validated": False,
        "data_persistent": False, "runtime_action_override": False,
        "internal_diagnostic_only": True, "test_fixture": False,
        "bindings": checked["bindings"], "artifacts": checked["artifacts"],
        "gates": checked["gates"], "sources": checked["sources"],
        "package_contract": checked["package_contract"], "self_path": relative,
    }
    _reject_sensitive(value, "diagnostic admission")
    _write_new(output_path, value)
    saved_sha = file_hash(output_path)
    try:
        read_saved_admission(output_path, expected_sha256=saved_sha, components=paths)
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    return value


def read_saved_admission(path: str | Path, *, expected_sha256: str,
                         components: Mapping[str, str | Path]) -> dict[str, Any]:
    saved_path, relative = _artifact_path(path, "diagnostic admission")
    if file_hash(saved_path) != _sha(expected_sha256, "diagnostic admission"):
        raise ValueError("Diagnostic admission bytes differ")
    value = _strict_json(saved_path, "diagnostic admission")
    if (set(value) != TOP_LEVEL_FIELDS or value.get("version") != VERSION
            or value.get("status") != STATUS or value.get("admitted") is not True
            or value.get("namespace") != NAMESPACE
            or value.get("pilot_class") != NAMESPACE
            or value.get("behavior_performance_gate_passed") is not False
            or value.get("behavior_performance_gate_waived") is not True
            or value.get("waiver_scope") != ["behavior_performance"]
            or value.get("formal_ready") is not False
            or value.get("formal_sample_eligible") is not False
            or value.get("human_explanation_effect_validated") is not False
            or value.get("data_persistent") is not False
            or value.get("runtime_action_override") is not False
            or value.get("internal_diagnostic_only") is not True
            or value.get("test_fixture") is not False
            or value.get("self_path") != relative
            or not isinstance(value.get("bindings"), dict)
            or set(value["bindings"]) != BINDING_FIELDS
            or value.get("gates") != {name: True for name in GATE_NAMES}):
        raise ValueError("Exact non-formal diagnostic admission required")
    checked = validate_components(components)
    for key in ("bindings", "artifacts", "gates", "sources", "package_contract"):
        if canonical(value.get(key)) != canonical(checked[key]):
            raise ValueError("Diagnostic admission " + key + " differs from live evidence")
    _reject_sensitive(value, "diagnostic admission")
    return deepcopy(value)


__all__ = [
    "VERSION", "STATUS", "NAMESPACE", "ARTIFACT_NAMES", "GATE_NAMES",
    "BINDING_FIELDS", "TOP_LEVEL_FIELDS", "source_closure", "package_contract",
    "validate_components", "build_admission", "read_saved_admission",
]
