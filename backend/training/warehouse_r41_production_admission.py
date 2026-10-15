"""Fail-closed production admission for the warehouse r4.1 internal pilot.

The admission is a technical evidence boundary, not a formal-study claim.  It
reopens and semantically validates every producer artifact, reconstructs the
frozen NumPy runtime, and refuses to write an admission until all registered
gates pass.  The portable package builder consumes only this record plus its
externally supplied SHA-256 and the exact bound component files.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

from backend.training import warehouse_r41_active_evaluation as active_evaluation
from backend.training import warehouse_r41_corrected_partner_audit as corrected_partner_audit
from backend.training import warehouse_r41_conflict_play_selection as play_selection
from backend.training import warehouse_r41_explanation_audit as explanation_audit
from backend.training import warehouse_r41_final_rcpd as final_rcpd
from backend.training import warehouse_r41_question_bank as question_bank
from backend.training import warehouse_r41_training_ledger as training_ledger
from backend.training import warehouse_r4_active_evaluation as original_evaluation
from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.warehouse_r41_online_explanation import R41OnlineAlignmentExplainer
from backend.warehouse_r41_online_runtime import R41OnlineAlignmentRuntime
from env.warehouse_native.policy import NumPyNativeActor
from env.warehouse_native.r41_conflict import r41_scene_fingerprint
from ui import warehouse_alignment_r41_tutorial as tutorial_api


ROOT = Path(__file__).resolve().parents[2]
VERSION = "warehouse-r41-production-admission.v1"
STATUS = "admitted_internal_pilot"
NAMESPACE = "internal_pilot"

ARTIFACT_NAMES = (
    "actor", "protocol", "training_ledger", "dual_evaluation",
    "corrected_six_partner_audit_report",
    "conflict_manifest", "conflict_validation", "task_conflict_graph",
    "dynamic_selection_report", "selected_scenes", "final_rcpd_report",
    "final_rcpd_program", "explanation_audit_report", "question_bank",
    "question_bank_report", "tutorial",
)
GATE_NAMES = (
    "training_budget_and_earliest_actor", "dual_original_and_conflict_evaluation",
    "corrected_six_partner_boundary_audit",
    "conflict_manifest_and_successor_contract", "dynamic_six_scene_selection",
    "post_freeze_independent_rcpd", "explanation_and_intervention_audit",
    "frozen_question_bank", "neutral_tutorial_replay", "runtime_action_authority",
    "source_and_package_contract",
)
BINDING_FIELDS = frozenset((
    "actor_sha256", "actor_parameters_sha256", "protocol_file_sha256",
    "protocol_sha256", "training_ledger_sha256", "dual_evaluation_sha256",
    "corrected_six_partner_audit_sha256",
    "conflict_manifest_file_sha256", "conflict_manifest_content_sha256",
    "conflict_validation_sha256", "conflict_contract_sha256",
    "conflict_graph_sha256", "dynamic_selection_report_sha256",
    "selected_scenes_file_sha256", "selected_scenes_sha256",
    "final_rcpd_report_sha256", "final_rcpd_binding_sha256",
    "program_sha256", "explanation_audit_sha256", "runtime_signature",
    "explainer_signature", "question_bank_sha256", "question_bank_signature",
    "question_bank_report_sha256", "tutorial_sha256", "tutorial_signature",
    "tutorial_scene_id", "tutorial_scene_fingerprint",
    "tutorial_successor_state_sha256", "tutorial_snapshot_sha256",
    "runtime_audit_sha256", "release_sources_sha256",
    "package_contract_sha256",
))
TOP_LEVEL_FIELDS = frozenset((
    "version", "status", "admitted", "namespace", "formal_ready",
    "internal_pilot_only", "human_explanation_effect_validated",
    "test_fixture", "bindings", "artifacts", "gates", "sources",
    "package_contract", "self_path",
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


def _strict_json(path: Path, label: str) -> dict[str, Any]:
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or path.stat().st_size > 512 * 1024 * 1024):
        raise ValueError(label + " is missing, linked, noncanonical, or oversized")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON field in " + label)
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError("Non-finite JSON value in " + label + ": " + token)))
    if not isinstance(value, dict):
        raise ValueError(label + " must be a JSON object")
    return value


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise ValueError("Explicit lowercase SHA-256 required for " + label)
    return value


def _artifact_path(value: str | Path, label: str) -> tuple[Path, str]:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.absolute()
    if path.resolve() != path or path.is_symlink() or not path.is_file():
        raise ValueError(label + " must be a canonical regular file")
    try:
        relative = path.relative_to(ROOT).as_posix()
    except ValueError:
        raise ValueError(label + " must stay inside the repository") from None
    if not _RELATIVE.fullmatch(relative) or ".." in Path(relative).parts:
        raise ValueError(label + " has an unsafe repository path")
    return path, relative


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
    """Return the exact transitive admission/extraction/release source set."""
    from backend.training.warehouse_r4_production_admission import local_source_hashes
    seeds = (
        Path(__file__),
        ROOT / "scripts/build_warehouse_r41_admission.py",
        ROOT / "scripts/run_warehouse_r41_postfreeze_release.py",
        ROOT / "scripts/build_warehouse_r41_online_release.py",
        ROOT / "scripts/build_warehouse_r41_neutral_tutorial.py",
        Path(training_ledger.__file__), Path(active_evaluation.__file__),
        Path(corrected_partner_audit.__file__),
        Path(play_selection.__file__), Path(final_rcpd.__file__),
        Path(explanation_audit.__file__), Path(question_bank.__file__),
        ROOT / "backend/warehouse_r41_online_runtime.py",
        ROOT / "backend/warehouse_r41_online_explanation.py",
        ROOT / "ui/warehouse_alignment_r41_online_release.py",
        Path(tutorial_api.__file__),
    )
    return local_source_hashes(seeds)


def package_contract() -> dict[str, Any]:
    from ui import warehouse_alignment_r41_online_release as release
    return {
        "release_version": release.VERSION,
        "release_status": release.STATUS,
        "admission_version": VERSION,
        "artifact_paths": deepcopy(release.ARTIFACT_PATHS),
        "archive_whitelist": sorted(release.ARCHIVE_WHITELIST),
        "maximum_package_bytes": release.MAX_PACKAGE_BYTES,
        "maximum_base64_bytes": release.MAX_BASE64_BYTES,
        "release_sources_sha256": digest(release.release_sources()),
        "requires_external_admission_sha256": True,
        "independent_final_program": True,
        "formal_ready": False,
    }


def _validate_scene_publication(manifest: Mapping[str, Any], manifest_path: Path,
                                validation: Mapping[str, Any], validation_path: Path,
                                graph: Mapping[str, Any], graph_path: Path) -> None:
    from backend.training.warehouse_r41_conflict_scenarios import (
        validate_conflict_manifest,
    )
    recomputed = validate_conflict_manifest(manifest, replay=True)
    required = {
        "version", "passed", "manifest_content_sha256", "manifest_file_sha256",
        "contract_sha256", "conflict_graph_sha256", "candidate_count",
        "operational_counts", "successor_replay_enabled", "split_seed_disjoint",
        "split_fingerprint_disjoint", "split_successor_state_disjoint",
        "old_geometry_reused", "forbidden_seed_reused",
        "spawned_on_agent_endpoint", "spawned_on_agent_endpoint_total",
        "task_conflict_graph_file_sha256",
    }
    if (set(validation) != required
            or validation.get("version")
                != "warehouse-r41-conflict-manifest-validation.v1"
            or validation.get("passed") is not True
            or validation.get("manifest_file_sha256") != file_hash(manifest_path)
            or validation.get("task_conflict_graph_file_sha256") != file_hash(graph_path)
            or digest(graph) != manifest.get("conflict_graph_sha256")):
        raise ValueError("Exact passing r4.1 conflict publication required")
    for key, value in recomputed.items():
        if validation.get(key) != value:
            raise ValueError("R4.1 conflict validation differs: " + key)


def _revalidate_ledger(path: Path, expected_sha256: str) -> dict[str, Any]:
    saved = training_ledger.read_saved_ledger(
        path, expected_sha256=expected_sha256, require_selected=True,
    )
    selected = saved["selected"]
    selected_actor = Path(selected["actor_path"]).expanduser().resolve()
    boundary = selected_actor.parent
    if boundary.parent.name != "boundaries":
        raise ValueError("R4.1 selected Actor is not in a committed boundary")
    run_root = boundary.parent.parent
    with tempfile.TemporaryDirectory(prefix="warehouse-r41-ledger-recheck-") as tmp:
        rebuilt_path = Path(tmp) / "ledger.json"
        rebuilt = training_ledger.build(run_root, rebuilt_path)
    if canonical(rebuilt) != canonical(saved):
        raise ValueError("R4.1 ledger differs from reconstructed run evidence")
    return saved


def _without_elapsed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(value))
    result.pop("elapsed_seconds", None)
    return result


def _revalidate_dual(dual: Mapping[str, Any], *, actor_path: Path,
                     manifest: Mapping[str, Any], manifest_path: Path,
                     source_actor_sha256: str) -> None:
    """Physically rerun all four Actor/suite matrices behind the dual report."""
    manifests = dual.get("manifests", {})
    original_info = manifests.get("original", {})
    conflict_info = manifests.get("conflict", {})
    original_path, _ = _artifact_path(original_info.get("path", ""),
                                      "r4.1 original validation manifest")
    if (file_hash(original_path) != original_info.get("file_sha256")
            or conflict_info.get("path") != str(manifest_path)
            or conflict_info.get("file_sha256") != file_hash(manifest_path)
            or conflict_info.get("semantic_sha256") != digest(manifest)):
        raise ValueError("R4.1 dual evaluation manifest bindings differ")
    original = _strict_json(original_path, "r4.1 original validation manifest")
    original_scenes = original.get("splits", {}).get("validation")
    conflict_scenes = manifest.get("splits", {}).get("conflict_validation")
    if (not isinstance(original_scenes, list) or len(original_scenes) != 50
            or not isinstance(conflict_scenes, list) or len(conflict_scenes) < 50
            or digest(original_scenes) != original_info.get("validation_entries_sha256")
            or digest(conflict_scenes) != conflict_info.get("validation_entries_sha256")):
        raise ValueError("R4.1 dual evaluation suite membership differs")
    baseline_receipt = dual.get("original", {}).get("baseline", {})
    baseline_path, _ = _artifact_path(
        baseline_receipt.get("actor", {}).get("path", ""), "r4.1 baseline Actor")
    if file_hash(baseline_path) != source_actor_sha256:
        raise ValueError("R4.1 dual evaluation uses another baseline Actor")
    matrices = (
        ("original", "baseline", original_evaluation.evaluate_actor,
         baseline_path, original_scenes),
        ("original", "candidate", original_evaluation.evaluate_actor,
         actor_path, original_scenes),
        ("conflict", "baseline", active_evaluation.evaluate_conflict_actor,
         baseline_path, conflict_scenes),
        ("conflict", "candidate", active_evaluation.evaluate_conflict_actor,
         actor_path, conflict_scenes),
    )
    with tempfile.TemporaryDirectory(prefix="warehouse-r41-dual-recheck-") as tmp:
        for suite, role, evaluator, candidate, scenes in matrices:
            stored = dual.get(suite, {}).get(role)
            if not isinstance(stored, dict):
                raise ValueError("R4.1 dual evaluation matrix is incomplete")
            replayed = evaluator(candidate, scenes,
                                 output_dir=Path(tmp) / suite / role,
                                 seed=stored.get("seed"))
            if canonical(_without_elapsed(replayed)) != canonical(
                    _without_elapsed(stored)):
                raise ValueError("R4.1 dual evaluation differs from physical replay")
    decisions = {
        "original_validation": active_evaluation._suite_decision(
            dual["original"]["baseline"]["summary"],
            dual["original"]["candidate"]["summary"]),
        "conflict_validation": active_evaluation._suite_decision(
            dual["conflict"]["baseline"]["summary"],
            dual["conflict"]["candidate"]["summary"]),
    }
    selected = all(row["selected"] for row in decisions.values())
    if (dual.get("suite_decisions") != decisions or not selected
            or dual.get("selected") is not True or dual.get("status") != "passed"):
        raise ValueError("R4.1 dual evaluation gates differ from replayed summaries")


def _revalidate_corrected_partner_audit(
        path: Path, *, expected_sha256: str, ledger: Mapping[str, Any],
        ledger_sha256: str, actor_sha256: str, dual_evaluation_sha256: str,
        manifest: Mapping[str, Any], manifest_sha256: str) -> dict[str, Any]:
    """Require the independent replay of every committed boundary.

    This gate deliberately does not derive eligibility from the frozen
    runner's ``selected`` bit.  The additive producer evaluates all boundary
    Actors under the corrected six-partner protocol, and its strict reader
    recomputes every per-suite and earliest-selection decision.
    """
    corrected = corrected_partner_audit.read_saved_report(
        path, expected_sha256=expected_sha256,
        expected_ledger_sha256=ledger_sha256,
        expected_actor_sha256=actor_sha256, require_passed=True,
    )
    expected_steps = [row["step"] for row in ledger["boundaries"]]
    expected_actors = [row["actor_sha256"] for row in ledger["boundaries"]]
    if (corrected.get("dual_evaluation_sha256") != dual_evaluation_sha256
            or corrected.get("conflict_manifest_sha256") != manifest_sha256
            or corrected.get("conflict_manifest_content_sha256")
                != manifest["content_sha256"]
            or corrected.get("evaluated_boundary_steps") != expected_steps
            or [row.get("actor_sha256") for row in corrected.get("boundaries", [])]
                != expected_actors
            or corrected.get("evaluated_boundary_count") != len(expected_steps)
            or corrected.get("all_ledger_boundaries_replayed") is not True
            or corrected.get("selected_actor_shutdown_count") != 0):
        raise ValueError(
            "Corrected six-partner audit is not bound to every ledger boundary")
    return corrected


def _runtime_audit(runtime: R41OnlineAlignmentRuntime,
                   selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    scenes = [selection["tutorial"], *selection["X"], *selection["Y"]]
    if (len(scenes) != 7 or len({row.get("id") for row in scenes}) != 7
            or len({row.get("fingerprint") for row in scenes[1:]}) != 6):
        raise ValueError("R4.1 runtime requires one tutorial and six unique play scenes")
    rows = []
    for scene in scenes:
        env = runtime.environment(deepcopy(scene))
        before = digest(env.snapshot())
        if r41_scene_fingerprint(env) != scene["fingerprint"]:
            raise ValueError("R4.1 runtime restored another scene")
        actions, decision = runtime.decision(env)
        if (digest(env.snapshot()) != before
                or decision.get("post_policy_overrides") != 0
                or actions["robot_2"] != decision["policy_actions"]["robot_2"]):
            raise ValueError("R4.1 runtime decision changed state or action")
        branch = runtime.counterfactual(env.snapshot(), ["WAIT"], steps=1)
        if digest(env.snapshot()) != before or branch.get("steps_executed") != 1:
            raise ValueError("R4.1 runtime counterfactual is not isolated")
        transition = runtime.step(env, "WAIT")
        if (transition["submitted_actions"]["robot_2"]
                != transition["policy_actions"]["robot_2"]
                or transition["decision"]["post_policy_overrides"] != 0):
            raise ValueError("R4.1 runtime overrode the frozen Actor")
        rows.append({
            "id": scene["id"], "fingerprint": scene["fingerprint"],
            "source_state_sha256": before,
            "policy_action": actions["robot_2"],
            "submitted_action": transition["submitted_actions"]["robot_2"],
            "source_unchanged_by_decision": True,
            "counterfactual_isolated": True,
            "post_policy_overrides": 0,
        })
    return rows


def _validate_question_report(report: Mapping[str, Any], payload: Mapping[str, Any],
                              manifest: Mapping[str, Any], manifest_file_sha: str,
                              runtime: R41OnlineAlignmentRuntime) -> None:
    expected_fields = {
        "version", "status", "actor_sha256", "protocol_sha256",
        "runtime_signature", "pool_manifest_sha256", "candidate_frames",
        "selected_scenarios", "checks", "source_bank_signature",
        "payload_sha256", "formal_ready", "conflict_manifest_content_sha256",
        "conflict_contract_sha256", "conflict_graph_sha256",
    }
    if (set(report) != expected_fields or report.get("version") != question_bank.VERSION
            or report.get("status") != "candidate_ready"
            or report.get("actor_sha256") != runtime.actor_sha256
            or report.get("protocol_sha256") != runtime.protocol_sha256
            or report.get("runtime_signature") != runtime.signature
            or report.get("pool_manifest_sha256") != manifest_file_sha
            or report.get("selected_scenarios") != 8
            or report.get("checks") != payload.get("checks")
            or report.get("source_bank_signature")
                != payload.get("source_bank_signature")
            or report.get("payload_sha256") != digest(payload)
            or report.get("formal_ready") is not False
            or report.get("conflict_manifest_content_sha256")
                != manifest["content_sha256"]
            or report.get("conflict_contract_sha256") != manifest["contract_sha256"]
            or report.get("conflict_graph_sha256") != manifest["conflict_graph_sha256"]
            or type(report.get("candidate_frames")) is not int
            or report["candidate_frames"] < 8):
        raise ValueError("R4.1 frozen question-bank report differs")


def validate_components(paths: Mapping[str, str | Path]) -> dict[str, Any]:
    """Load every component and recompute all technical admission gates."""
    if not isinstance(paths, Mapping) or set(paths) != set(ARTIFACT_NAMES):
        raise ValueError("Exact r4.1 production artifact set required")
    files, relatives = {}, {}
    for name in ARTIFACT_NAMES:
        files[name], relatives[name] = _artifact_path(paths[name], name)
    hashes = {name: file_hash(path) for name, path in files.items()}

    protocol = _strict_json(files["protocol"], "r4.1 protocol")
    ledger = _revalidate_ledger(files["training_ledger"], hashes["training_ledger"])
    selected = ledger["selected"]
    if (Path(selected["actor_path"]).resolve() != files["actor"]
            or selected["actor_sha256"] != hashes["actor"]
            or ledger["protocol_sha256"] != hashes["protocol"]
            or ledger["protocol_semantic_sha256"] != digest(protocol)
            or Path(selected["evaluation_path"]).resolve() != files["dual_evaluation"]
            or selected["evaluation_sha256"] != hashes["dual_evaluation"]):
        raise ValueError("R4.1 ledger, Actor, protocol or dual evaluation differs")
    dual = _strict_json(files["dual_evaluation"], "r4.1 dual evaluation")
    if (dual.get("version") != active_evaluation.VERSION
            or dual.get("status") != "passed" or dual.get("selected") is not True
            or dual.get("candidate_actor_sha256") != hashes["actor"]
            or dual.get("action_authority_exact") is not True):
        raise ValueError("R4.1 dual validation is not a passing exact-Actor audit")

    manifest = _strict_json(files["conflict_manifest"], "r4.1 conflict manifest")
    validation = _strict_json(files["conflict_validation"], "r4.1 conflict validation")
    graph = _strict_json(files["task_conflict_graph"], "r4.1 conflict graph")
    _validate_scene_publication(
        manifest, files["conflict_manifest"], validation,
        files["conflict_validation"], graph, files["task_conflict_graph"],
    )
    if (ledger.get("conflict_manifest_semantic_sha256") != manifest["content_sha256"]
            or ledger.get("conflict_contract_sha256") != manifest["contract_sha256"]
            or ledger.get("conflict_graph_sha256") != manifest["conflict_graph_sha256"]):
        raise ValueError("R4.1 ledger uses another conflict publication")
    _revalidate_dual(
        dual, actor_path=files["actor"], manifest=manifest,
        manifest_path=files["conflict_manifest"],
        source_actor_sha256=ledger["source"]["actor_sha256"],
    )
    corrected = _revalidate_corrected_partner_audit(
        files["corrected_six_partner_audit_report"],
        expected_sha256=hashes["corrected_six_partner_audit_report"],
        ledger=ledger, ledger_sha256=hashes["training_ledger"],
        actor_sha256=hashes["actor"],
        dual_evaluation_sha256=hashes["dual_evaluation"],
        manifest=manifest, manifest_sha256=hashes["conflict_manifest"],
    )

    dynamic = play_selection.read_saved_report(
        files["dynamic_selection_report"].parent,
        expected_report_sha256=hashes["dynamic_selection_report"],
        actor_path=files["actor"], manifest_path=files["conflict_manifest"],
        selected_scenes_path=files["selected_scenes"], require_selected=True,
    )
    selection = _strict_json(files["selected_scenes"], "r4.1 selected scenes")
    if (selection.get("actor_sha256") != hashes["actor"]
            or dynamic.get("release_eligible") is not True):
        raise ValueError("R4.1 dynamic selection uses another Actor")

    final = final_rcpd.read_saved_report(
        files["final_rcpd_report"].parent,
        expected_report_sha256=hashes["final_rcpd_report"],
        actor_path=files["actor"], scenarios_path=files["conflict_manifest"],
        program_path=files["final_rcpd_program"], require_passed=True,
    )
    if (final.get("status") != "passed"
            or final.get("program_file_sha256") != hashes["final_rcpd_program"]
            or final.get("execution", {}).get("ppo_joint_steps") != 0
            or final.get("execution", {}).get("optimizer_updates") != 0
            or final.get("execution", {}).get("program_feedback_into_actor") is not False):
        raise ValueError("R4.1 post-freeze RCPD is not independently qualified")

    runtime = R41OnlineAlignmentRuntime(
        files["actor"], protocol=protocol,
        expected_actor_sha256=hashes["actor"],
        expected_protocol_sha256=digest(protocol), allow_test_fixture=False,
    )
    runtime.verify_binding()
    explainer = R41OnlineAlignmentExplainer(
        files["final_rcpd_program"],
        expected_program_sha256=hashes["final_rcpd_program"],
        runtime=runtime, allow_test_fixture=False,
    )
    explainer._assert_current(runtime)
    heldout = manifest["splits"]["final_test"]
    explanation_bindings = {
        "actor_sha256": runtime.actor_sha256,
        "protocol_sha256": runtime.protocol_sha256,
        "runtime_signature": runtime.signature,
        "program_sha256": explainer.program_sha256,
        "explainer_signature": explainer.signature,
        "scenario_manifest_sha256": digest(manifest),
        "test_fixture": False,
        "contract_sha256": digest(explanation_audit.contract()),
        "producer_sources_sha256": digest(explanation_audit.producer_sources()),
        "holdout_fingerprints_sha256": digest(sorted(
            scene["fingerprint"] for scene in heldout)),
    }
    explanation = explanation_audit.read_saved_report(
        files["explanation_audit_report"].parent,
        expected_report_sha256=hashes["explanation_audit_report"],
        expected_bindings=explanation_bindings,
    )
    if explanation.get("status") != "passed":
        raise ValueError("R4.1 explanation and intervention audit did not pass")

    question = _strict_json(files["question_bank"], "r4.1 question bank")
    question_bank.validate_payload(runtime, question)
    question_report = _strict_json(files["question_bank_report"],
                                   "r4.1 question-bank report")
    _validate_question_report(question_report, question, manifest,
                              hashes["conflict_manifest"], runtime)

    tutorial = _strict_json(files["tutorial"], "r4.1 neutral tutorial")
    tutorial_scene = selection["tutorial"]
    tutorial_result = tutorial_api.validate_neutral_tutorial(
        tutorial, tutorial_scene=tutorial_scene, runtime=runtime,
        expected_bindings={
            "scene_manifest_version": manifest["version"],
            "scene_manifest_content_sha256": manifest["content_sha256"],
            "tutorial_scene_fingerprint": tutorial_scene["fingerprint"],
            "tutorial_successor_state_sha256": tutorial_scene["snapshot"][
                "r41_conflict"]["successor_state_sha256"],
            "tutorial_snapshot_sha256": digest(tutorial_scene["snapshot"]),
            "conflict_contract_sha256": manifest["contract_sha256"],
            "conflict_graph_sha256": manifest["conflict_graph_sha256"],
            "producer_sources_sha256": digest(tutorial_api.producer_sources()),
        },
    )
    tutorial_signature = digest(tutorial)
    if (tutorial_result.get("tutorial_signature") != tutorial_signature
            or tutorial.get("uses_final_actor") is not False):
        raise ValueError("R4.1 neutral tutorial boundary differs")

    runtime_rows = _runtime_audit(runtime, selection)
    sources = source_closure()
    package = package_contract()
    for label, value in (
        ("protocol", protocol), ("ledger", ledger), ("dual evaluation", dual),
        ("corrected six-partner audit", corrected),
        ("scene manifest", manifest), ("dynamic selection", selection),
        ("final RCPD", final), ("explanation audit", explanation),
        ("question bank", question), ("tutorial", tutorial),
    ):
        _reject_sensitive(value, label)
    metadata = NumPyNativeActor(files["actor"]).metadata
    bindings = {
        "actor_sha256": hashes["actor"],
        "actor_parameters_sha256": metadata["actor_parameters_sha256"],
        "protocol_file_sha256": hashes["protocol"],
        "protocol_sha256": digest(protocol),
        "training_ledger_sha256": hashes["training_ledger"],
        "dual_evaluation_sha256": hashes["dual_evaluation"],
        "corrected_six_partner_audit_sha256": hashes[
            "corrected_six_partner_audit_report"],
        "conflict_manifest_file_sha256": hashes["conflict_manifest"],
        "conflict_manifest_content_sha256": manifest["content_sha256"],
        "conflict_validation_sha256": hashes["conflict_validation"],
        "conflict_contract_sha256": manifest["contract_sha256"],
        "conflict_graph_sha256": manifest["conflict_graph_sha256"],
        "dynamic_selection_report_sha256": hashes["dynamic_selection_report"],
        "selected_scenes_file_sha256": hashes["selected_scenes"],
        "selected_scenes_sha256": digest(selection),
        "final_rcpd_report_sha256": hashes["final_rcpd_report"],
        "final_rcpd_binding_sha256": final["final_rcpd_binding_sha256"],
        "program_sha256": hashes["final_rcpd_program"],
        "explanation_audit_sha256": hashes["explanation_audit_report"],
        "runtime_signature": runtime.signature,
        "explainer_signature": explainer.signature,
        "question_bank_sha256": hashes["question_bank"],
        "question_bank_signature": question["source_bank_signature"],
        "question_bank_report_sha256": hashes["question_bank_report"],
        "tutorial_sha256": hashes["tutorial"],
        "tutorial_signature": tutorial_signature,
        "tutorial_scene_id": tutorial_scene["id"],
        "tutorial_scene_fingerprint": tutorial_scene["fingerprint"],
        "tutorial_successor_state_sha256": tutorial_scene["snapshot"][
            "r41_conflict"]["successor_state_sha256"],
        "tutorial_snapshot_sha256": digest(tutorial_scene["snapshot"]),
        "runtime_audit_sha256": digest(runtime_rows),
        "release_sources_sha256": package["release_sources_sha256"],
        "package_contract_sha256": digest(package),
    }
    if set(bindings) != BINDING_FIELDS:
        raise AssertionError("Internal r4.1 admission binding schema differs")
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
        raise ValueError("R4.1 admission output path is unsafe")
    raw = (canonical(value) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                         | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def build_admission(paths: Mapping[str, str | Path], *, output: str | Path) -> dict[str, Any]:
    output_path = Path(output).expanduser()
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path = output_path.absolute()
    try:
        relative = output_path.relative_to(ROOT).as_posix()
    except ValueError:
        raise ValueError("R4.1 admission output must stay inside the repository") from None
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    checked = validate_components(paths)
    admission = {
        "version": VERSION, "status": STATUS, "admitted": True,
        "namespace": NAMESPACE, "formal_ready": False,
        "internal_pilot_only": True,
        "human_explanation_effect_validated": False,
        "test_fixture": False,
        "bindings": checked["bindings"], "artifacts": checked["artifacts"],
        "gates": checked["gates"], "sources": checked["sources"],
        "package_contract": checked["package_contract"],
        "self_path": relative,
    }
    _reject_sensitive(admission, "r4.1 admission")
    _write_new(output_path, admission)
    admission_sha = file_hash(output_path)
    try:
        read_saved_admission(output_path, expected_sha256=admission_sha,
                             components=paths)
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    return admission


def read_saved_admission(path: str | Path, *, expected_sha256: str,
                         components: Mapping[str, str | Path]) -> dict[str, Any]:
    """Authenticate and fully revalidate one persisted admission."""
    path, relative = _artifact_path(path, "r4.1 production admission")
    if file_hash(path) != _sha(expected_sha256, "r4.1 production admission"):
        raise ValueError("R4.1 production admission bytes differ")
    admission = _strict_json(path, "r4.1 production admission")
    if (set(admission) != TOP_LEVEL_FIELDS or admission.get("version") != VERSION
            or admission.get("status") != STATUS or admission.get("admitted") is not True
            or admission.get("namespace") != NAMESPACE
            or admission.get("formal_ready") is not False
            or admission.get("internal_pilot_only") is not True
            or admission.get("human_explanation_effect_validated") is not False
            or admission.get("test_fixture") is not False
            or admission.get("self_path") != relative
            or not isinstance(admission.get("bindings"), dict)
            or set(admission["bindings"]) != BINDING_FIELDS
            or admission.get("gates") != {name: True for name in GATE_NAMES}):
        raise ValueError("Exact non-formal r4.1 production admission required")
    checked = validate_components(components)
    for key in ("bindings", "artifacts", "gates", "sources", "package_contract"):
        if canonical(admission.get(key)) != canonical(checked[key]):
            raise ValueError("R4.1 admission " + key + " differs from live evidence")
    _reject_sensitive(admission, "r4.1 admission")
    return deepcopy(admission)


__all__ = [
    "VERSION", "STATUS", "NAMESPACE", "ARTIFACT_NAMES", "GATE_NAMES",
    "BINDING_FIELDS", "TOP_LEVEL_FIELDS", "source_closure", "package_contract",
    "validate_components", "build_admission", "read_saved_admission",
]
