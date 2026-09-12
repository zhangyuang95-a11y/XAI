"""Fail-closed admission for the warehouse r4.1 diagnostic v8 release.

The only waived gate is the already disclosed behaviour-performance gate. A
release cannot be admitted until the frozen public-tree candidate, its single
fresh-final audit, the six-family play selection, questionnaire and neutral
tutorial all authenticate against the designated terminal Actor.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from pathlib import PureWindowsPath
import re
from typing import Any, Mapping, Sequence

from backend.training import warehouse_r41_diagnostic_conflict_play_selection as selection_api
from backend.training import warehouse_r41_diagnostic_conflict_scenarios as scenes_api
from backend.training import warehouse_r41_diagnostic_designation as designation_api
from backend.training import warehouse_r41_diagnostic_development_expansion_v8 as expansion_api
from backend.training import warehouse_r41_diagnostic_explanation_audit_v8 as explanation_api
from backend.training import warehouse_r41_diagnostic_final_once_v8 as final_once_api
from backend.training import warehouse_r41_diagnostic_fresh_final_holdout_v3 as retired_v3_api
from backend.training import warehouse_r41_diagnostic_fresh_final_holdout_v4 as holdout_api
from backend.training import warehouse_r41_diagnostic_pair_weights_v8 as pair_weight_api
from backend.training import warehouse_r41_diagnostic_question_bank as question_api
from backend.training import warehouse_r41_diagnostic_rcpd_v8 as rcpd_api
from backend.training.warehouse_native_common import canonical, digest, file_hash
from backend.warehouse_r41_diagnostic_boosted_tree import VERSION as BOOSTED_TREE_VERSION
from backend.warehouse_r41_diagnostic_online_explanation_v8 import R41DiagnosticOnlineAlignmentExplainer
from backend.warehouse_r41_diagnostic_online_runtime import R41DiagnosticOnlineAlignmentRuntime
from backend.warehouse_r41_diagnostic_public_features_v8 import R41DiagnosticPublicRelationsV8
from backend.warehouse_r41_diagnostic_public_tree_program_v8 import (
    AGGREGATION, GROUPS, VERSION as PUBLIC_TREE_VERSION,
    R41DiagnosticPublicTreeProgramV8,
)
from env.warehouse_native.policy import NumPyNativeActor
from env.warehouse_native.r41_diagnostic_conflict import (
    CONFLICT_FAMILIES_SHA256, DIAGNOSTIC_CONTRACT_SHA256,
    diagnostic_contract_receipt, diagnostic_scene_fingerprint,
)
from ui import warehouse_alignment_r41_diagnostic_tutorial as tutorial_api


ROOT = Path(__file__).resolve().parents[2]
VERSION = "warehouse-r41-diagnostic-admission.v6"
STATUS = "admitted_internal_diagnostic"
NAMESPACE = "internal_diagnostic"
FIXED_ACTOR_SHA256 = "4ac2ba7782b5556761edaab22bfad50c831c1d8b41b174245e2d81486287ff6b"
EXACT_ACTION_NAMES = ("UP", "DOWN", "LEFT", "RIGHT", "WAIT")
EXACT_CLASSES = (0, 1, 2, 3, 4)
MAX_PROGRAM_BYTES = 64 * 1024 * 1024
MAX_COMPONENT_ITERATIONS = 512
MAX_TOTAL_ITERATIONS = 2048
MAX_TOTAL_TREES = 10_240
MAX_TOTAL_NODES = 2_500_000
MAXIMUM_TREE_DEPTH = 16
EXPECTED_RETIRED_HOLDOUTS = {
    "warehouse-r41-diagnostic-fresh-final-holdout.v1": {
        "file_sha256": (
            "6c3ff25f9917451815979173d98881fe5e5e98258b514d199d50ee362d2746c2"
        ),
        "content_sha256": (
            "92d3a1320a08dcfa07e8a68e2b8e43f8d3748072e3c887a965c9b7efd9cfe627"
        ),
    },
    "warehouse-r41-diagnostic-fresh-final-holdout.v2": {
        "file_sha256": (
            "7d72424744eea7547516cffe414c15578802c7204b90bdf1b129a1e9acfbf17b"
        ),
        "content_sha256": (
            "a0072c07e196ceca2680fbdde2f786d4867a1862ee8461a7fc943cbc8f765e28"
        ),
    },
}

ARTIFACT_NAMES = (
    "diagnostic_designation", "actor", "protocol", "training_ledger",
    "dual_evaluation", "failure_closeout", "diagnostic_contract",
    "conflict_manifest", "conflict_validation", "dynamic_selection_report",
    "selected_scenes", "development_supplement",
    "development_expansion_registry", "development_expansion_report",
    "prior_v7_rcpd_report", "prior_v7_rcpd_rows",
    "development_expansion_rows", "v8_fit_config", "final_rcpd_report",
    "final_rcpd_rows", "final_rcpd_program",
    "retired_fresh_final_holdout_v1", "retired_fresh_final_holdout_v2",
    "final_once_attempt_started", "final_once_candidate_authenticated",
    "final_once_holdout_started", "final_once_holdout_completed",
    "final_once_audit_started", "final_once_audit_completed",
    "final_once_attempt_completed", "fresh_final_holdout",
    "fresh_final_holdout_report", "fresh_final_v3_exclusion",
    "explanation_audit_inputs",
    "explanation_audit_evidence", "explanation_audit_report",
    "physical_replay", "question_bank", "question_bank_report", "tutorial",
)
FINAL_ONCE_LEDGER_ARTIFACTS = {
    "final_once_attempt_started": "attempt_started.json",
    "final_once_candidate_authenticated": "candidate_authenticated.json",
    "final_once_holdout_started": "holdout_started.json",
    "final_once_holdout_completed": "holdout_completed.json",
    "final_once_audit_started": "audit_started.json",
    "final_once_audit_completed": "audit_completed.json",
    "final_once_attempt_completed": "attempt_completed.json",
}
FINAL_ONCE_CANDIDATE_AUTH_FIELDS = frozenset((
    "version", "status", "campaign_key", "candidate_identity_sha256",
    "attempt_started_sha256", "candidate_artifacts",
    "candidate_artifacts_sha256", "rcpd_report_file_sha256",
    "program_file_sha256", "rows_file_sha256",
    "prior_v7_report_file_sha256", "prior_v7_rows_file_sha256",
    "expansion_rows_file_sha256", "fit_config_file_sha256",
    "actor_file_sha256", "protocol_file_sha256", "manifest_file_sha256",
    "designation_file_sha256", "selected_scenes_file_sha256",
    "development_registries", "require_passed", "refit", "formal_ready",
))
GATE_NAMES = (
    "terminal_actor_designation", "runtime_action_authority",
    "diagnostic_conflict_publication", "dynamic_six_family_scene_selection",
    "program_blind_development_expansion", "frozen_public_tree_v8",
    "one_shot_fresh_final_audit", "heldout_explanation_and_intervention_audit",
    "independent_physical_replay", "frozen_bilingual_question_bank",
    "neutral_tutorial_replay", "source_and_portable_package_contract",
)
BINDING_FIELDS = frozenset((
    "diagnostic_designation_sha256", "actor_sha256", "actor_parameters_sha256",
    "actor_metadata_sha256", "protocol_file_sha256", "protocol_content_sha256",
    "training_ledger_sha256", "dual_evaluation_sha256", "failure_closeout_sha256",
    "diagnostic_contract_file_sha256", "diagnostic_contract_semantic_sha256",
    "diagnostic_contract_sha256", "conflict_graph_sha256", "conflict_families_sha256",
    "conflict_manifest_file_sha256", "conflict_manifest_content_sha256",
    "conflict_manifest_semantic_sha256", "conflict_validation_sha256",
    "dynamic_selection_report_sha256", "selected_scenes_file_sha256",
    "selected_scenes_semantic_sha256", "development_supplement_sha256",
    "development_expansion_registry_sha256", "development_expansion_content_sha256",
    "development_expansion_report_sha256", "prior_v7_rcpd_report_sha256",
    "prior_v7_rcpd_rows_sha256", "development_expansion_rows_sha256",
    "v8_fit_config_sha256", "final_rcpd_report_sha256", "final_rcpd_rows_sha256",
    "final_rcpd_binding_sha256", "fresh_final_holdout_sha256",
    "fresh_final_holdout_content_sha256", "fresh_final_holdout_report_sha256",
    "fresh_final_v3_exclusion_sha256",
    "fresh_final_v3_exclusion_content_sha256",
    "final_once_identity_sha256", "final_once_campaign_key",
    "final_once_permanent_anchor_sha256", "final_once_attempt_started_sha256",
    "final_once_candidate_authenticated_sha256",
    "final_once_holdout_started_sha256", "final_once_holdout_completed_sha256",
    "final_once_audit_started_sha256", "final_once_audit_completed_sha256",
    "final_once_attempt_completed_sha256", "program_sha256",
    "program_content_sha256", "program_identity_sha256",
    "public_feature_contract_sha256", "public_feature_registry_sha256",
    "program_complexity_sha256", "explanation_audit_inputs_sha256",
    "explanation_audit_evidence_sha256", "explanation_audit_sha256",
    "physical_replay_sha256", "runtime_signature", "runtime_manifest_signature",
    "explainer_signature", "question_bank_sha256", "question_bank_signature",
    "question_bank_report_sha256", "tutorial_sha256", "tutorial_signature",
    "tutorial_scene_id", "tutorial_scene_fingerprint",
    "tutorial_successor_state_sha256", "tutorial_snapshot_sha256",
    "runtime_audit_sha256", "release_sources_sha256", "package_contract_sha256",
))
TOP_LEVEL_FIELDS = frozenset((
    "version", "status", "admitted", "namespace", "pilot_class",
    "behavior_performance_gate_passed", "behavior_performance_gate_waived",
    "waiver_scope", "formal_ready", "formal_sample_eligible",
    "human_explanation_effect_validated", "data_persistent",
    "runtime_action_override", "internal_diagnostic_only", "test_fixture",
    "bindings", "artifacts", "gates", "sources", "package_contract", "self_path",
))

_HEX = re.compile(r"[0-9a-f]{64}\Z")
_RELATIVE = re.compile(r"[A-Za-z0-9_.\-/]+\Z")
_SENSITIVE_KEY = re.compile(r"(?:^|[_-])(?:api[_-]?key|secret(?:[_-]?key)?|password|passwd|database[_-]?url|authorization|bearer[_-]?token)(?:$|[_-])", re.I)
_SENSITIVE_VALUE = re.compile(r"(?:\bsk-[A-Za-z0-9_-]{16,}|\bBearer\s+[A-Za-z0-9._-]{12,}|\b(?:postgres(?:ql)?|mongodb(?:\+srv)?|mysql)://|(?:^|[/\\])Users[/\\])", re.I)


def _sha(value: Any, label: str) -> str:
    if type(value) is not str or _HEX.fullmatch(value) is None:
        raise ValueError("Exact lowercase SHA-256 required for " + label)
    return value


def _strict_json(path: Path, label: str, *, maximum: int = 512 * 1024 * 1024) -> dict[str, Any]:
    if (not path.is_file() or path.is_symlink() or path.resolve() != path
            or not 0 < path.stat().st_size <= maximum):
        raise ValueError(label + " is missing, linked, noncanonical, empty, or oversized")
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result:
                raise ValueError("Duplicate JSON field in " + label)
            result[key] = value
        return result
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError("Non-finite JSON value in " + label + ": " + token)))
    except UnicodeDecodeError as error:
        raise ValueError(label + " must be UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(label + " must be one JSON object")
    return value


def _artifact_path(value: str | Path, label: str, *,
                   external_name: str | None = None) -> tuple[Path, str]:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.absolute()
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ValueError(label + " must be a canonical regular file")
    if external_name is not None:
        if (path.name != external_name
                or path == ROOT.resolve() or ROOT.resolve() in path.parents):
            raise ValueError(label + " must be the external final-once ledger file")
        relative = "external/final_once_ledger/" + external_name
    else:
        try:
            relative = path.relative_to(ROOT.resolve()).as_posix()
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
    elif isinstance(value, str):
        stripped = value.strip()
        if (_SENSITIVE_VALUE.search(value) or Path(stripped).is_absolute()
                or PureWindowsPath(stripped).is_absolute()):
            raise ValueError("Credential-like or absolute-local value in " + label)


def _default_routes() -> tuple[dict[str, Any], ...]:
    return tuple({"feature_name": "derived.critical." + group,
                  "operator": ">", "threshold": 0.5} for group in GROUPS)


def _final_claim_binding(*, campaign_key: str,
                         attempt_started_sha256: str,
                         candidate_identity_sha256: str,
                         candidate_authenticated_sha256: str) -> dict[str, str]:
    """Return the exact claim lineage written into the v4 v3 tombstone."""
    return {
        "campaign_key": _sha(campaign_key, "final-once campaign key"),
        "attempt_started_sha256": _sha(
            attempt_started_sha256, "final-once attempt start"),
        "candidate_identity_sha256": _sha(
            candidate_identity_sha256, "final-once candidate identity"),
        "candidate_authenticated_sha256": _sha(
            candidate_authenticated_sha256,
            "final-once candidate authentication"),
    }


def _require_v3_claim_binding(value: Any,
                              expected: Mapping[str, str]) -> None:
    if (not isinstance(value, Mapping) or set(value) != set(expected)
            or dict(value) != dict(expected)):
        raise ValueError("Fresh-final v3 exclusion claim binding differs")


def _audit_runtime_source_bindings(
        runtime: R41DiagnosticOnlineAlignmentRuntime) -> dict[str, Any]:
    """Mirror the exact runtime/source identity emitted by audit v8."""
    full_manifest = deepcopy(runtime.source_full_manifest_bindings)
    runtime_sources = explanation_api.diagnostic_runtime_sources()
    return {
        "source_full_manifest_bindings": full_manifest,
        "source_full_manifest_bindings_sha256": digest(full_manifest),
        "runtime_sources": runtime_sources,
        "runtime_sources_sha256": digest(runtime_sources),
    }


def _audit_candidate_bindings(
        *, candidate_authenticated_sha256: str,
        candidate_artifacts: Mapping[str, str]) -> dict[str, Any]:
    """Mirror audit v8's post-claim strict-candidate lineage fields."""
    artifacts = {str(name): _sha(value, "candidate artifact " + str(name))
                 for name, value in sorted(candidate_artifacts.items())}
    if set(artifacts) != set(final_once_api.CANDIDATE_ARTIFACT_KEYS):
        raise ValueError("Exact final-once candidate artifact registry required")
    return {
        "candidate_authenticated_sha256": _sha(
            candidate_authenticated_sha256,
            "final-once candidate authentication"),
        "candidate_artifacts": artifacts,
        "candidate_artifacts_sha256": digest(artifacts),
    }


def _require_exact_producer_source_closure(
        value: Any, expected: Mapping[str, str], *, label: str) -> None:
    """Require the complete recursive producer closure and every file hash."""
    if (not isinstance(value, Mapping) or dict(value) != dict(expected)
            or any(type(name) is not str or _HEX.fullmatch(str(item)) is None
                   for name, item in value.items())):
        raise ValueError(label + " producer source closure differs")


def _require_exact_audit_bindings(value: Any,
                                  expected: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping) or dict(value) != dict(expected):
        raise ValueError("Final explanation-audit external bindings differ")


def _validate_candidate_authentication_marker(
        marker: Mapping[str, Any], *, files: Mapping[str, Path],
        hashes: Mapping[str, str], singles: Mapping[str, Path],
        row_paths: Sequence[Path], development_paths: Sequence[Path],
        campaign_key: str, candidate_identity_sha256: str,
) -> dict[str, str]:
    """Bind post-claim final use to the exact strict-refit candidate bytes."""
    candidate_artifacts = final_once_api._validate_candidate_artifact_binding(
        marker, singles=singles, row_paths=row_paths)
    development_registries: dict[str, str] = {}
    for path in development_paths:
        version = _strict_json(path, "development registry").get("version")
        if type(version) is not str or version in development_registries:
            raise ValueError("Candidate authentication development registry differs")
        development_registries[version] = file_hash(path)
    expected_scalar = {
        "version": final_once_api.VERSION + ".candidate-authentication.v1",
        "status": "passed_strict_reader_and_refit",
        "campaign_key": campaign_key,
        "candidate_identity_sha256": candidate_identity_sha256,
        "attempt_started_sha256": hashes["final_once_attempt_started"],
        "actor_file_sha256": hashes["actor"],
        "protocol_file_sha256": hashes["protocol"],
        "manifest_file_sha256": hashes["conflict_manifest"],
        "designation_file_sha256": hashes["diagnostic_designation"],
        "selected_scenes_file_sha256": hashes["selected_scenes"],
        "development_registries": dict(sorted(development_registries.items())),
        "require_passed": True,
        "refit": True,
        "formal_ready": False,
    }
    if (set(marker) != FINAL_ONCE_CANDIDATE_AUTH_FIELDS
            or any(marker.get(name) != expected
                   for name, expected in expected_scalar.items())
            or marker.get("candidate_artifacts") != candidate_artifacts
            or marker.get("candidate_artifacts_sha256")
                != digest(candidate_artifacts)
            or file_hash(files["final_once_candidate_authenticated"])
                != hashes["final_once_candidate_authenticated"]):
        raise ValueError("Exact strict-refit candidate authentication marker required")
    return candidate_artifacts


def validate_v8_program_identity(program_path: str | Path, *,
        actor_feature_names: Sequence[str], actor_sha256: str,
        actor_parameters_sha256: str,
        source_full_manifest_bindings: Mapping[str, Any],
        rcpd_report: Mapping[str, Any], fit_config: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate the public program independently of its producer."""
    path = Path(program_path).expanduser().absolute()
    if (path.suffix != ".json" or path.is_symlink() or not path.is_file()
            or path.resolve() != path or path.stat().st_size > MAX_PROGRAM_BYTES):
        raise ValueError("v8 program must be one bounded canonical JSON file")
    raw = path.read_bytes()
    if raw.startswith(b"\x80") or b"pickle" in raw.lower() or b"joblib" in raw.lower():
        raise ValueError("Pickle/joblib program serialization is forbidden")
    payload = _strict_json(path, "v8 public-tree program", maximum=MAX_PROGRAM_BYTES)
    program = R41DiagnosticPublicTreeProgramV8.from_dict(payload)
    relations = R41DiagnosticPublicRelationsV8(actor_feature_names)
    config = rcpd_api.normalize_config(fit_config)
    expected_metadata = {
        "native_source_actor_sha256": actor_sha256,
        "source_actor_parameters_sha256": actor_parameters_sha256,
        "source_full_manifest_bindings": dict(source_full_manifest_bindings),
        "diagnostic_rcpd_version": rcpd_api.VERSION,
        "diagnostic_rcpd_binding_sha256": rcpd_report.get("diagnostic_rcpd_binding_sha256"),
        "fit_config_sha256": digest(config),
        "public_feature_contract_sha256": digest(relations.contract()),
        "pair_weight_contract_sha256": digest(pair_weight_api.contract()),
        "actions": list(EXACT_ACTION_NAMES), "classes": list(EXACT_CLASSES),
        "runtime_controller": "native_neural_actor_only",
        "runtime_action_override": False, "program_feedback_into_actor": False,
        "formal_ready": False,
    }
    if (program.action_names != EXACT_ACTION_NAMES
            or program.base_program.classes != EXACT_CLASSES
            or program.base_feature_names != tuple(actor_feature_names)
            or program.feature_names != relations.feature_names
            or program.routes != _default_routes()
            or program.mix_weights != config["mix_weights"]
            or payload.get("aggregation") != AGGREGATION
            or set(program.metadata) != set(expected_metadata)
            or any(program.metadata.get(key) != value for key, value in expected_metadata.items())
            or program.metadata.get("action_legality_features") not in (None, {}, [])
            or program.metadata.get("action_constraint_reason_features") not in (None, {}, [])):
        raise ValueError("v8 public program external identity differs")
    for name, component in (("base", program.base_program),
            *((group, program._specialist_programs[group]) for group in GROUPS)):
        metadata = component.metadata
        expected_component_metadata = {
            "diagnostic_rcpd_version": rcpd_api.VERSION,
            "diagnostic_rcpd_binding_sha256": rcpd_report.get(
                "diagnostic_rcpd_binding_sha256"),
            "component": name,
            "fit_rows": metadata.get("fit_rows"),
            "fit_config": config["models"][name],
            "prediction_input": "349 deterministic public features",
            "validation_labels_used_for_fit": False,
            "actor_logits_used_as_program_input": False,
            "actor_hidden_states_used": False,
            "runtime_action_override": False,
        }
        if (component.classes != EXACT_CLASSES or component.action_names != EXACT_ACTION_NAMES
                or component.feature_names != relations.feature_names
                or set(metadata) != set(expected_component_metadata)
                or type(metadata.get("fit_rows")) is not int
                or metadata.get("fit_rows", 0) <= 0
                or any(metadata.get(key) != value
                       for key, value in expected_component_metadata.items())):
            raise ValueError(name + " public-tree component identity differs")
    complexity = program.complexity()
    if (complexity != rcpd_report.get("candidate", {}).get("complexity")
            or complexity["component_count"] != 4 or complexity["specialist_count"] != 3
            or complexity["route_count"] != 3
            or complexity["maximum_component_iterations"] > MAX_COMPONENT_ITERATIONS
            or complexity["total_iterations"] > MAX_TOTAL_ITERATIONS
            or complexity["total_trees"] > MAX_TOTAL_TREES
            or complexity["total_nodes"] > MAX_TOTAL_NODES
            or complexity["maximum_tree_depth"] > MAXIMUM_TREE_DEPTH
            or complexity["unique_split_features"] > len(relations.feature_names)):
        raise ValueError("v8 public program complexity differs or exceeds release limits")
    if (file_hash(path) != rcpd_report.get("program_file_sha256")
            or digest(payload) != rcpd_report.get("program_content_sha256")
            or digest(payload) != rcpd_report.get("candidate", {}).get("program_content_sha256")):
        raise ValueError("v8 public program bytes/content binding differs")
    _reject_sensitive(payload, "v8 public-tree program")
    identity = {
        "version": PUBLIC_TREE_VERSION, "component_version": BOOSTED_TREE_VERSION,
        "actions": list(EXACT_ACTION_NAMES), "classes": list(EXACT_CLASSES),
        "relations": relations.contract(), "routes": list(_default_routes()),
        "mix_weights": program.mix_weights, "aggregation": AGGREGATION,
        "metadata": program.metadata, "complexity": complexity,
    }
    return {"program_file_sha256": file_hash(path),
            "program_content_sha256": digest(payload),
            "program_identity_sha256": digest(identity),
            "public_feature_contract_sha256": digest(relations.contract()),
            "public_feature_registry_sha256": digest(list(relations.feature_names)),
            "program_complexity_sha256": digest(complexity), "complexity": complexity}


def _diagnostic_runtime(actor: Path, protocol_path: Path, manifest_path: Path,
        protocol: Mapping[str, Any], manifest: Mapping[str, Any]):
    content = deepcopy(dict(manifest)); claimed = content.pop("content_sha256", None)
    runtime = R41DiagnosticOnlineAlignmentRuntime(actor,
        training_protocol_path=protocol_path, manifest_path=manifest_path,
        expected_actor_sha256=FIXED_ACTOR_SHA256,
        expected_training_protocol_file_sha256=file_hash(protocol_path),
        expected_training_protocol_content_sha256=digest(protocol),
        expected_manifest_file_sha256=file_hash(manifest_path),
        expected_manifest_content_sha256=claimed,
        expected_manifest_semantic_sha256=digest(manifest))
    _reject_sensitive(runtime.actor.metadata, "diagnostic Actor metadata")
    runtime.verify_binding(); return runtime


def _validate_publication(files: Mapping[str, Path], hashes: Mapping[str, str],
        manifest: Mapping[str, Any]) -> None:
    receipt = _strict_json(files["diagnostic_contract"], "diagnostic contract")
    if (canonical(receipt) != canonical(diagnostic_contract_receipt())
            or receipt.get("contract_sha256") != DIAGNOSTIC_CONTRACT_SHA256
            or receipt.get("conflict_families_sha256") != CONFLICT_FAMILIES_SHA256):
        raise ValueError("Diagnostic conflict contract receipt differs")
    recomputed = scenes_api.validate_diagnostic_manifest(manifest, replay=True,
        workload_actor_path=files["actor"], replay_workloads=True)
    saved = _strict_json(files["conflict_validation"], "conflict validation")
    expected = {**recomputed, "artifacts": {
        "diagnostic_contract.json": hashes["diagnostic_contract"],
        "manifest.json": hashes["conflict_manifest"]}}
    if saved != expected or saved.get("passed") is not True:
        raise ValueError("Diagnostic scene publication differs from physical replay")


def _validate_expansion(files: Mapping[str, Path], hashes: Mapping[str, str],
        actor: NumPyNativeActor, manifest: Mapping[str, Any]) -> dict[str, Any]:
    expansion_dir = _same_parent(
        files,
        ("development_expansion_registry", "development_expansion_report"),
        "development expansion",
    )
    if (files["development_expansion_registry"]
            != expansion_dir / "development_expansion.json"
            or files["development_expansion_report"] != expansion_dir / "report.json"):
        raise ValueError("Development-expansion evidence layout differs")
    registry = _strict_json(files["development_expansion_registry"], "development expansion")
    report = _strict_json(files["development_expansion_report"], "development expansion report")
    content = {key: value for key, value in registry.items() if key != "content_sha256"}
    report_content = {key: value for key, value in report.items() if key != "content_sha256"}
    if (registry.get("version") != expansion_api.VERSION or registry.get("status") != expansion_api.STATUS
            or registry.get("content_sha256") != digest(content)
            or registry.get("contract") != expansion_api.contract()
            or any(registry.get(key) is not False for key in (
                "program_access", "program_predictions_access", "final_audit_rows_access",
                "final_labels_used_for_selection", "runtime_action_override"))
            or report.get("version") != expansion_api.VERSION or report.get("status") != expansion_api.STATUS
            or report.get("content_sha256") != digest(report_content)
            or report.get("registry_file_sha256") != hashes["development_expansion_registry"]
            or report.get("registry_content_sha256") != registry["content_sha256"]
            or report.get("bindings") != registry.get("bindings")
            or registry.get("bindings", {}).get("actor_sha256") != FIXED_ACTOR_SHA256
            or registry.get("bindings", {}).get("actor_parameters_sha256")
                != actor.metadata["actor_parameters_sha256"]
            or registry.get("bindings", {}).get("source_manifest_file_sha256")
                != hashes["conflict_manifest"]
            or registry.get("bindings", {}).get("source_manifest_content_sha256")
                != manifest.get("content_sha256")):
        raise ValueError("Exact program-blind development expansion required")
    return registry


def _validate_retired_holdout_bindings(
        files: Mapping[str, Path], hashes: Mapping[str, str],
        expansion: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    actual: dict[str, dict[str, str]] = {}
    for index in (1, 2):
        name = f"retired_fresh_final_holdout_v{index}"
        value = _strict_json(files[name], "retired fresh-final holdout")
        version = value.get("version")
        content = {key: child for key, child in value.items()
                   if key != "content_sha256"}
        claimed_content = value.get("content_sha256")
        if (version not in EXPECTED_RETIRED_HOLDOUTS or version in actual
                or claimed_content != digest(content)):
            raise ValueError("Retired fresh-final holdout identity differs")
        actual[str(version)] = {
            "file_sha256": hashes[name],
            "content_sha256": claimed_content,
        }
    expansion_bindings = expansion.get("bindings")
    expansion_retired = (expansion_bindings.get("retired_holdouts")
                         if isinstance(expansion_bindings, Mapping) else None)
    if (actual != EXPECTED_RETIRED_HOLDOUTS
            or expansion_retired != EXPECTED_RETIRED_HOLDOUTS):
        raise ValueError(
            "Retired v1/v2 bytes must match the fixed development expansion")
    return deepcopy(actual)


def _validate_bilingual_question_bank(payload: Mapping[str, Any]) -> None:
    items = payload.get("items")
    if not isinstance(items, list) or len(items) != 8:
        raise ValueError("Exact frozen eight-item question bank required")
    for item in items:
        prompt = item.get("prompt") if isinstance(item, Mapping) else None
        options = item.get("options") if isinstance(item, Mapping) else None
        if (not isinstance(prompt, Mapping) or set(prompt) != {"zh", "en"}
                or any(type(prompt[key]) is not str or not prompt[key].strip() for key in ("zh", "en"))
                or not isinstance(options, list) or not options):
            raise ValueError("Question bank must contain complete Chinese and English text")
        for option in options:
            label = option.get("label") if isinstance(option, Mapping) else None
            if (not isinstance(label, Mapping) or set(label) != {"zh", "en"}
                    or any(type(label[key]) is not str or not label[key].strip() for key in ("zh", "en"))):
                raise ValueError("Question option must contain complete Chinese and English text")


def _runtime_audit(runtime: R41DiagnosticOnlineAlignmentRuntime,
        selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    scenes = [selection["tutorial"], *selection["X"], *selection["Y"]]
    if (len(scenes) != 7 or len({row.get("id") for row in scenes}) != 7
            or len({row.get("family_id") for row in scenes[1:]}) != 6):
        raise ValueError("Diagnostic runtime requires one tutorial and six families")
    result = []; player_actions = ("WAIT", "UP", "WAIT", "LEFT", "WAIT", "DOWN", "WAIT", "RIGHT")
    for scene in scenes:
        env = runtime.environment(deepcopy(scene))
        if diagnostic_scene_fingerprint(env) != scene.get("fingerprint"):
            raise ValueError("Diagnostic runtime restored another scene")
        initial = digest(env.snapshot())
        branch = runtime.counterfactual(env.snapshot(), ["WAIT"], steps=1)
        if digest(env.snapshot()) != initial or branch.get("steps_executed") != 1:
            raise ValueError("Diagnostic runtime counterfactual is not isolated")
        steps = 0
        while not env.done:
            actions, decision = runtime.decision(env)
            if (decision.get("post_policy_overrides") != 0
                    or actions["robot_2"] != decision["policy_actions"]["robot_2"]):
                raise ValueError("Diagnostic runtime changed the neural action")
            transition = runtime.step(env, player_actions[steps % len(player_actions)])
            steps += 1
            if (transition["submitted_actions"]["robot_2"] != transition["policy_actions"]["robot_2"]
                    or transition["submitted_actions"]["robot_2"] != actions["robot_2"]):
                raise ValueError("Diagnostic runtime overrode the frozen Actor")
            if steps > runtime.config.horizon:
                raise ValueError("Diagnostic runtime exceeded its public horizon")
        result.append({"id": scene["id"], "fingerprint": scene["fingerprint"],
            "family_id": scene.get("family_id"), "initial_state_sha256": initial,
            "steps": steps, "terminal_state_sha256": digest(env.snapshot()),
            "counterfactual_isolated": True, "post_policy_overrides": 0,
            "all_policy_actions_submitted_exactly": True})
    return result


def source_closure() -> dict[str, str]:
    from backend.training.warehouse_r4_production_admission import local_source_hashes
    return local_source_hashes((Path(__file__),
        ROOT / "scripts/build_warehouse_r41_diagnostic_admission_v6.py",
        Path(designation_api.__file__), Path(scenes_api.__file__), Path(selection_api.__file__),
        Path(expansion_api.__file__), Path(rcpd_api.__file__), Path(pair_weight_api.__file__),
        Path(final_once_api.__file__), Path(holdout_api.__file__),
        Path(retired_v3_api.__file__), Path(explanation_api.__file__),
        Path(question_api.__file__), ROOT / "backend/warehouse_r41_diagnostic_online_runtime.py",
        ROOT / "backend/warehouse_r41_diagnostic_online_explanation_v8.py",
        ROOT / "backend/warehouse_r41_diagnostic_public_features_v8.py",
        ROOT / "backend/warehouse_r41_diagnostic_public_tree_program_v8.py",
        ROOT / "backend/warehouse_r41_diagnostic_boosted_tree.py", Path(tutorial_api.__file__),
        ROOT / "ui/warehouse_alignment_r41_diagnostic_release_v8.py",
        ROOT / "scripts/build_warehouse_r41_diagnostic_online_release_v8.py",
        ROOT / "scripts/preflight_warehouse_r41_diagnostic_render_v6.py",
        ROOT / "backend/training/warehouse_r41_diagnostic_release_receipt_v6.py",
        ROOT / "scripts/build_warehouse_r41_diagnostic_release_receipt_v6.py"))


def package_contract() -> dict[str, Any]:
    from ui import warehouse_alignment_r41_diagnostic_release_v8 as release
    return {"release_version": release.VERSION,
        "public_release_version": release.PUBLIC_RELEASE_VERSION,
        "release_status": release.STATUS, "admission_version": VERSION,
        "artifact_paths": deepcopy(release.ARTIFACT_PATHS),
        "archive_whitelist": sorted(release.ARCHIVE_WHITELIST),
        "maximum_package_bytes": release.MAX_PACKAGE_BYTES,
        "maximum_base64_bytes": release.MAX_BASE64_BYTES,
        "archive_compression": "ZIP_BZIP2",
        "archive_compresslevel": release.ARCHIVE_COMPRESSLEVEL,
        "release_sources_sha256": digest(release.release_sources()),
        "requires_external_admission_sha256": True,
        "uses_compact_seven_scene_runtime_manifest": True,
        "source_full_manifest_remains_hash_bound": True,
        "program_serialization": "explicit_json_axis_threshold_trees",
        "pickle_allowed": False, "behavior_performance_waiver_scope": ["behavior_performance"],
        "runtime_action_override": False, "data_persistent": False,
        "formal_ready": False, "formal_sample_eligible": False}


def _same_parent(files: Mapping[str, Path], names: Sequence[str], label: str) -> Path:
    parents = {files[name].parent for name in names}
    if len(parents) != 1:
        raise ValueError(label + " artifacts must share one directory")
    return next(iter(parents))


def _require_fixed_final_once_registry(registry: Path, campaign_key: str) -> None:
    expected = final_once_api._ledger_root() / _sha(
        campaign_key, "final-once campaign key")
    if registry != expected:
        raise ValueError("Final-once permanent registry layout differs")


def validate_components(paths: Mapping[str, str | Path]) -> dict[str, Any]:
    if not isinstance(paths, Mapping) or set(paths) != set(ARTIFACT_NAMES):
        raise ValueError("Exact r4.1 diagnostic v8 admission artifact set required")
    files, relatives = {}, {}
    for name in ARTIFACT_NAMES:
        files[name], relatives[name] = _artifact_path(
            paths[name], name,
            external_name=FINAL_ONCE_LEDGER_ARTIFACTS.get(name),
        )
    hashes = {name: file_hash(path) for name, path in files.items()}
    if hashes["actor"] != FIXED_ACTOR_SHA256:
        raise ValueError("Exact user-designated terminal Actor required")
    designation_components = {name: files[name] for name in designation_api.ARTIFACT_NAMES}
    designation = designation_api.read_saved_designation(files["diagnostic_designation"],
        expected_sha256=hashes["diagnostic_designation"], components=designation_components)
    if (designation.get("behavior_performance_gate_passed") is not False
            or designation.get("behavior_performance_gate_waived") is not True
            or designation.get("waiver_scope") != ["behavior_performance"]
            or designation.get("formal_ready") is not False
            or designation.get("formal_sample_eligible") is not False
            or designation.get("runtime_action_override") is not False
            or designation.get("bindings", {}).get("actor_sha256") != FIXED_ACTOR_SHA256):
        raise ValueError("Exact diagnostic Actor designation required")
    protocol = _strict_json(files["protocol"], "training protocol")
    manifest = _strict_json(files["conflict_manifest"], "conflict manifest")
    _validate_publication(files, hashes, manifest)
    runtime = _diagnostic_runtime(files["actor"], files["protocol"],
        files["conflict_manifest"], protocol, manifest)
    actor = NumPyNativeActor(files["actor"])
    dynamic = selection_api.read_saved_diagnostic_selection(
        files["dynamic_selection_report"].parent,
        expected_report_sha256=hashes["dynamic_selection_report"],
        actor_path=files["actor"], manifest_path=files["conflict_manifest"],
        selected_scenes_path=files["selected_scenes"], require_selected=True)
    selection = _strict_json(files["selected_scenes"], "selected scenes")
    if (dynamic.get("passed") is not True or selection.get("release_eligible") is not True
            or selection.get("six_distinct_conflict_families") is not True
            or selection.get("zero_action_overrides") is not True
            or selection.get("ordinary_sampler_fallback") is not False):
        raise ValueError("Exact passing six-family scene selection required")
    expansion = _validate_expansion(files, hashes, actor, manifest)
    _validate_retired_holdout_bindings(files, hashes, expansion)
    rcpd_dir = _same_parent(files, ("prior_v7_rcpd_report", "prior_v7_rcpd_rows",
        "development_expansion_rows", "v8_fit_config", "final_rcpd_report",
        "final_rcpd_rows", "final_rcpd_program"), "v8 RCPD")
    expected_names = {"prior_v7_rcpd_report": "prior_v7_report.json",
        "prior_v7_rcpd_rows": "prior_v7_rows.npz",
        "development_expansion_rows": "expansion_rows.npz", "v8_fit_config": "fit_config.json",
        "final_rcpd_report": "report.json", "final_rcpd_rows": "rows.npz",
        "final_rcpd_program": "program.json"}
    if any(files[name] != rcpd_dir / filename for name, filename in expected_names.items()):
        raise ValueError("v8 RCPD evidence layout differs")
    final = rcpd_api.read_saved_report(rcpd_dir,
        expected_report_sha256=hashes["final_rcpd_report"], actor_path=files["actor"],
        protocol_path=files["protocol"], manifest_path=files["conflict_manifest"],
        designation_path=files["diagnostic_designation"],
        expansion_registry_path=files["development_expansion_registry"],
        expected_expansion_registry_sha256=hashes["development_expansion_registry"],
        expected_prior_v7_report_sha256=hashes["prior_v7_rcpd_report"],
        previous_development_path=files["development_supplement"],
        expected_expansion_rows_sha256=hashes["development_expansion_rows"],
        expected_config_sha256=hashes["v8_fit_config"], require_passed=True, refit=False)
    program_identity = validate_v8_program_identity(files["final_rcpd_program"],
        actor_feature_names=actor.metadata["feature_names"], actor_sha256=FIXED_ACTOR_SHA256,
        actor_parameters_sha256=actor.metadata["actor_parameters_sha256"],
        source_full_manifest_bindings=runtime.source_full_manifest_bindings,
        rcpd_report=final, fit_config=_strict_json(files["v8_fit_config"], "v8 fit config"))
    holdout_dir = _same_parent(files, (
        "fresh_final_holdout", "fresh_final_holdout_report",
        "fresh_final_v3_exclusion"), "fresh-final")
    if files["fresh_final_v3_exclusion"] != holdout_dir / "v3_exclusion.json":
        raise ValueError("Fresh-final v3 exclusion layout differs")
    holdout = holdout_api.read_saved_holdout(holdout_dir,
        expected_holdout_sha256=hashes["fresh_final_holdout"],
        expected_report_sha256=hashes["fresh_final_holdout_report"])
    holdout_sources = holdout_api.producer_sources()
    holdout_report = _strict_json(
        files["fresh_final_holdout_report"], "fresh-final report")
    _require_exact_producer_source_closure(
        holdout_report.get("producer_sources"), holdout_sources,
        label="Fresh-final")
    if (holdout_report.get("bindings", {}).get("producer_sources_sha256")
            != digest(holdout_sources)):
        raise ValueError("Fresh-final producer source binding differs")
    if (holdout.get("status") != "passed_program_blind_registry"
            or holdout.get("program_access") is not False
            or holdout.get("program_predictions_access") is not False
            or holdout.get("final_labels_used_for_selection") is not False
            or holdout.get("statistics", {}).get("public_observation_overlap") != 0
            or holdout.get("bindings", {}).get("actor_sha256") != FIXED_ACTOR_SHA256):
        raise ValueError("Exact program-blind v4 fresh final holdout required")
    retired = [files[f"retired_fresh_final_holdout_v{index}"] for index in (1, 2)]
    singles, dev_paths, row_paths, retired_paths = final_once_api._input_paths(
        actor_path=files["actor"], protocol_path=files["protocol"],
        program_path=files["final_rcpd_program"], rcpd_report_path=files["final_rcpd_report"],
        manifest_path=files["conflict_manifest"], designation_path=files["diagnostic_designation"],
        selected_scenes_path=files["selected_scenes"],
        development_registry_paths=[files["development_supplement"], files["development_expansion_registry"]],
        development_rows_paths=[files["final_rcpd_rows"]], retired_holdout_paths=retired)
    expected_identity = final_once_api._preclaim_identity(singles, dev_paths)
    final_sources = final_once_api.producer_sources()
    final_campaign_key = final_once_api.campaign_key()
    if final_campaign_key != digest(expected_identity):
        raise ValueError("Final-once campaign identity differs")
    candidate_marker = _strict_json(
        files["final_once_candidate_authenticated"],
        "strict candidate authentication phase")
    candidate_artifacts = _validate_candidate_authentication_marker(
        candidate_marker, files=files, hashes=hashes, singles=singles,
        row_paths=row_paths, development_paths=dev_paths,
        campaign_key=final_campaign_key,
        candidate_identity_sha256=digest(expected_identity),
    )
    v3_exclusion = _strict_json(
        files["fresh_final_v3_exclusion"], "fresh-final v3 exclusion")
    v3_content = dict(v3_exclusion)
    claimed_v3_content = v3_content.pop("content_sha256", None)
    expected_claim = _final_claim_binding(
        campaign_key=final_campaign_key,
        attempt_started_sha256=hashes["final_once_attempt_started"],
        candidate_identity_sha256=digest(expected_identity),
        candidate_authenticated_sha256=hashes[
            "final_once_candidate_authenticated"],
    )
    _require_v3_claim_binding(v3_exclusion.get("claim_binding"), expected_claim)
    if (set(v3_exclusion) != {
            "version", "status", "legacy_selection_version",
            "legacy_selection_source_sha256",
            "legacy_selection_contract_sha256", "claim_binding", "scenes",
            "selection_trace", "statistics", "program_access",
            "program_predictions_access",
            "final_labels_used_for_program_selection",
            "runtime_action_override", "formal_ready", "content_sha256"}
            or v3_exclusion.get("version") != holdout_api.V3_TOMBSTONE_VERSION
            or v3_exclusion.get("status") != "burned_program_blind_exclusion"
            or v3_exclusion.get("legacy_selection_version")
                != retired_v3_api.VERSION
            or v3_exclusion.get("legacy_selection_source_sha256")
                != file_hash(Path(retired_v3_api.__file__).resolve())
            or v3_exclusion.get("legacy_selection_contract_sha256")
                != digest(retired_v3_api.contract())
            or not isinstance(v3_exclusion.get("scenes"), list)
            or len(v3_exclusion["scenes"]) != retired_v3_api.TOTAL_SCENES
            or len({row.get("fingerprint") for row in v3_exclusion["scenes"]
                    if isinstance(row, Mapping)}) != retired_v3_api.TOTAL_SCENES
            or not isinstance(v3_exclusion.get("selection_trace"), list)
            or not isinstance(v3_exclusion.get("statistics"), Mapping)
            or v3_exclusion["statistics"].get("accepted")
                != retired_v3_api.TOTAL_SCENES
            or v3_exclusion["statistics"].get("families")
                != dict(sorted(retired_v3_api.FAMILY_QUOTAS.items()))
            or v3_exclusion.get("program_access") is not False
            or v3_exclusion.get("program_predictions_access") is not False
            or v3_exclusion.get("final_labels_used_for_program_selection") is not False
            or v3_exclusion.get("runtime_action_override") is not False
            or v3_exclusion.get("formal_ready") is not False
            or claimed_v3_content != digest(v3_content)
            or holdout.get("bindings", {}).get("v3_exclusion_file_sha256")
                != hashes["fresh_final_v3_exclusion"]
            or holdout.get("bindings", {}).get("v3_exclusion_content_sha256")
                != claimed_v3_content):
        raise ValueError("Exact claim-bound fresh-final v3 exclusion required")
    ledger_names = (
        "final_once_attempt_started", "final_once_candidate_authenticated",
        "final_once_holdout_started", "final_once_holdout_completed",
        "final_once_audit_started", "final_once_audit_completed",
        "final_once_attempt_completed",
    )
    registry = _same_parent(files, ledger_names, "final-once registry")
    ledger_layout = {
        "attempt_started.json": "final_once_attempt_started",
        "candidate_authenticated.json": "final_once_candidate_authenticated",
        "holdout_started.json": "final_once_holdout_started",
        "holdout_completed.json": "final_once_holdout_completed",
        "audit_started.json": "final_once_audit_started",
        "audit_completed.json": "final_once_audit_completed",
        "attempt_completed.json": "final_once_attempt_completed",
    }
    if (files["final_once_attempt_started"] != registry / "attempt_started.json"
            or files["final_once_attempt_completed"] != registry / "attempt_completed.json"
            or any(files[name] != registry / relative
                   for relative, name in ledger_layout.items())
            or registry.name != final_campaign_key):
        raise ValueError("Final-once permanent registry layout differs")
    _require_fixed_final_once_registry(registry, final_campaign_key)
    completion = final_once_api.read_completion(registry,
        expected_completion_sha256=hashes["final_once_attempt_completed"],
        expected_identity=expected_identity)
    output_root = files["fresh_final_holdout"].parent.parent
    output_layout = {
        "fresh_holdout/v3_exclusion.json": "fresh_final_v3_exclusion",
        "fresh_holdout/holdout.json": "fresh_final_holdout",
        "fresh_holdout/report.json": "fresh_final_holdout_report",
        "explanation_audit/inputs.json": "explanation_audit_inputs",
        "explanation_audit/evidence.npz": "explanation_audit_evidence",
        "explanation_audit/report.json": "explanation_audit_report",
        "physical_replay.json": "physical_replay"}
    expected_completion_artifacts = {
        "attempt_started.json": hashes["final_once_attempt_started"],
        **{relative: hashes[name] for relative, name in output_layout.items()},
    }
    expected_phase_receipts = {
        relative: hashes[name]
        for relative, name in ledger_layout.items()
        if relative not in {"attempt_started.json", "attempt_completed.json"}
    }
    if (completion.get("version") != final_once_api.VERSION
            or completion.get("status") != "completed_passed"
            or completion.get("key") != final_campaign_key
            or completion.get("campaign_key") != final_campaign_key
            or completion.get("candidate_identity_sha256")
                != digest(expected_identity)
            or _HEX.fullmatch(str(
                completion.get("permanent_anchor_sha256"))) is None
            or completion.get("automatic_retry") is not False
            or completion.get("retry_allowed") is not False
            or completion.get("reason") is not None
            or completion.get("output_created") is not True
            or completion.get("holdout_status") != "passed_program_blind_registry"
            or completion.get("audit_status") != "passed"
            or completion.get("physical_replay_status") != "passed"
            or completion.get("program_fits") != 0 or completion.get("actor_updates") != 0
            or completion.get("candidate_authentication_status")
                != "passed_strict_reader_and_refit"
            or completion.get("candidate_authenticated_sha256")
                != hashes["final_once_candidate_authenticated"]
            or completion.get("candidate_artifacts") != candidate_artifacts
            or completion.get("candidate_artifacts_sha256")
                != digest(candidate_artifacts)
            or completion.get("development_authentication_refit") is not True
            or completion.get("phase_receipts") != expected_phase_receipts
            or completion.get("runtime_action_override") is not False
            or completion.get("formal_ready") is not False
            or completion.get("producer_sources") != final_sources
            or completion.get("output_identity") != str(output_root)
            or any(files[name] != output_root / relative for relative, name in output_layout.items())
            or completion.get("artifacts") != expected_completion_artifacts):
        raise ValueError("Exact completed passing one-shot final audit required")
    audit_dir = _same_parent(files, ("explanation_audit_inputs", "explanation_audit_evidence",
        "explanation_audit_report"), "explanation audit")
    audit_report = _strict_json(files["explanation_audit_report"], "explanation audit")
    audit_sources = explanation_api.producer_sources()
    audit_inputs = _strict_json(
        files["explanation_audit_inputs"], "explanation audit inputs")
    _require_exact_producer_source_closure(
        audit_inputs.get("producer_sources"), audit_sources,
        label="Explanation-audit")
    expected_audit_bindings = {"actor_sha256": FIXED_ACTOR_SHA256,
        "actor_parameters_sha256": actor.metadata["actor_parameters_sha256"],
        "protocol_file_sha256": hashes["protocol"], "protocol_content_sha256": digest(protocol),
        "manifest_file_sha256": hashes["conflict_manifest"], "manifest_semantic_sha256": digest(manifest),
        "designation_file_sha256": hashes["diagnostic_designation"],
        "expansion_registry_file_sha256": hashes["development_expansion_registry"],
        "program_file_sha256": hashes["final_rcpd_program"],
        "program_content_sha256": program_identity["program_content_sha256"],
        "rcpd_report_file_sha256": hashes["final_rcpd_report"],
        "fresh_holdout_file_sha256": hashes["fresh_final_holdout"],
        "fresh_holdout_content_sha256": holdout["content_sha256"],
        "fresh_holdout_report_file_sha256": hashes["fresh_final_holdout_report"],
        "development_rows": {files["final_rcpd_rows"].name: hashes["final_rcpd_rows"]},
        **_audit_candidate_bindings(
            candidate_authenticated_sha256=hashes[
                "final_once_candidate_authenticated"],
            candidate_artifacts=candidate_artifacts,
        ),
        "runtime_signature": runtime.signature,
        "runtime_manifest_signature": runtime.runtime_manifest_signature,
        **_audit_runtime_source_bindings(runtime),
        "holdout_fingerprints_sha256": digest(sorted(row["fingerprint"] for row in holdout["scenes"])),
        "contract_sha256": digest(explanation_api.contract()),
        "producer_sources_sha256": digest(audit_sources)}
    _require_exact_audit_bindings(
        audit_report.get("bindings"), expected_audit_bindings)
    explanation = explanation_api.read_saved_report(audit_dir,
        expected_report_sha256=hashes["explanation_audit_report"],
        program_path=files["final_rcpd_program"], development_rows_paths=[files["final_rcpd_rows"]],
        expected_bindings=expected_audit_bindings, require_passed=True)
    if (explanation.get("status") != "passed" or explanation.get("statistics", {}).get("passed") is not True
            or explanation.get("zero_nn_overrides") is not True
            or explanation.get("counterfactual_isolated") is not True
            or explanation.get("program_never_controls_action") is not True
            or explanation.get("raw_public_observations_persisted") is not True):
        raise ValueError("Exact passing heldout v8 explanation audit required")
    replay = _strict_json(files["physical_replay"], "physical replay")
    if (set(replay) != {"version", "status", "evidence_file_sha256",
                       "arrays_sha256", "execution", "program_fits",
                       "actor_updates", "runtime_action_override", "formal_ready"}
            or replay.get("version") != explanation_api.VERSION + ".physical-replay.v1"
            or replay.get("status") != "passed"
            or replay.get("evidence_file_sha256") != hashes["explanation_audit_evidence"]
            or _HEX.fullmatch(str(replay.get("arrays_sha256"))) is None
            or replay.get("execution") != explanation.get("execution", {}).get("counts")
            or replay.get("program_fits") != 0 or replay.get("actor_updates") != 0
            or replay.get("runtime_action_override") is not False or replay.get("formal_ready") is not False):
        raise ValueError("Exact independent physical replay receipt required")
    explainer = R41DiagnosticOnlineAlignmentExplainer(files["final_rcpd_program"],
        expected_program_sha256=hashes["final_rcpd_program"], runtime=runtime)
    explainer._assert_current(runtime)
    question = _strict_json(files["question_bank"], "question bank")
    _validate_bilingual_question_bank(question)
    question_report = question_api.read_saved_report(files["question_bank_report"].parent,
        expected_report_sha256=hashes["question_bank_report"],
        expected_question_bank_sha256=hashes["question_bank"], runtime=runtime,
        manifest=manifest, manifest_file_sha256=hashes["conflict_manifest"])
    if (question_report.get("status") != "candidate_ready"
            or question_report.get("actor_sha256") != FIXED_ACTOR_SHA256
            or question_report.get("checks") != question.get("checks")
            or question_report.get("payload_sha256") != digest(question)
            or question_report.get("formal_ready") is not False):
        raise ValueError("Frozen bilingual question-bank evidence differs")
    tutorial_scene = selection["tutorial"]
    tutorial = _strict_json(files["tutorial"], "neutral tutorial")
    expected_tutorial = tutorial_api.bindings(manifest, tutorial_scene,
        manifest_file_sha256=hashes["conflict_manifest"])
    tutorial_result = tutorial_api.validate_neutral_tutorial(tutorial,
        tutorial_scene=tutorial_scene, runtime=runtime, expected_bindings=expected_tutorial)
    tutorial_signature = digest(tutorial)
    if (tutorial_result.get("tutorial_signature") != tutorial_signature
            or tutorial.get("uses_final_actor") is not False):
        raise ValueError("Neutral tutorial boundary differs")
    runtime_rows = _runtime_audit(runtime, selection)
    package = package_contract(); sources = source_closure()
    for index, value in enumerate((designation, protocol, manifest, selection, expansion,
            final, holdout, explanation, replay, question, tutorial)):
        _reject_sensitive(value, "admission input " + str(index))
    contract_receipt = _strict_json(files["diagnostic_contract"], "diagnostic contract")
    bindings = {
        "diagnostic_designation_sha256": hashes["diagnostic_designation"],
        "actor_sha256": FIXED_ACTOR_SHA256,
        "actor_parameters_sha256": actor.metadata["actor_parameters_sha256"],
        "actor_metadata_sha256": digest(actor.metadata), "protocol_file_sha256": hashes["protocol"],
        "protocol_content_sha256": digest(protocol), "training_ledger_sha256": hashes["training_ledger"],
        "dual_evaluation_sha256": hashes["dual_evaluation"], "failure_closeout_sha256": hashes["failure_closeout"],
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
        "development_supplement_sha256": hashes["development_supplement"],
        "development_expansion_registry_sha256": hashes["development_expansion_registry"],
        "development_expansion_content_sha256": expansion["content_sha256"],
        "development_expansion_report_sha256": hashes["development_expansion_report"],
        "prior_v7_rcpd_report_sha256": hashes["prior_v7_rcpd_report"],
        "prior_v7_rcpd_rows_sha256": hashes["prior_v7_rcpd_rows"],
        "development_expansion_rows_sha256": hashes["development_expansion_rows"],
        "v8_fit_config_sha256": hashes["v8_fit_config"], "final_rcpd_report_sha256": hashes["final_rcpd_report"],
        "final_rcpd_rows_sha256": hashes["final_rcpd_rows"],
        "final_rcpd_binding_sha256": final["diagnostic_rcpd_binding_sha256"],
        "fresh_final_holdout_sha256": hashes["fresh_final_holdout"],
        "fresh_final_holdout_content_sha256": holdout["content_sha256"],
        "fresh_final_holdout_report_sha256": hashes["fresh_final_holdout_report"],
        "fresh_final_v3_exclusion_sha256": hashes[
            "fresh_final_v3_exclusion"],
        "fresh_final_v3_exclusion_content_sha256": claimed_v3_content,
        "final_once_identity_sha256": digest(expected_identity),
        "final_once_campaign_key": final_campaign_key,
        "final_once_permanent_anchor_sha256": completion[
            "permanent_anchor_sha256"],
        "final_once_attempt_started_sha256": hashes["final_once_attempt_started"],
        "final_once_candidate_authenticated_sha256": hashes[
            "final_once_candidate_authenticated"],
        "final_once_holdout_started_sha256": hashes[
            "final_once_holdout_started"],
        "final_once_holdout_completed_sha256": hashes[
            "final_once_holdout_completed"],
        "final_once_audit_started_sha256": hashes[
            "final_once_audit_started"],
        "final_once_audit_completed_sha256": hashes[
            "final_once_audit_completed"],
        "final_once_attempt_completed_sha256": hashes["final_once_attempt_completed"],
        "program_sha256": program_identity["program_file_sha256"],
        "program_content_sha256": program_identity["program_content_sha256"],
        "program_identity_sha256": program_identity["program_identity_sha256"],
        "public_feature_contract_sha256": program_identity["public_feature_contract_sha256"],
        "public_feature_registry_sha256": program_identity["public_feature_registry_sha256"],
        "program_complexity_sha256": program_identity["program_complexity_sha256"],
        "explanation_audit_inputs_sha256": hashes["explanation_audit_inputs"],
        "explanation_audit_evidence_sha256": hashes["explanation_audit_evidence"],
        "explanation_audit_sha256": hashes["explanation_audit_report"],
        "physical_replay_sha256": hashes["physical_replay"], "runtime_signature": runtime.signature,
        "runtime_manifest_signature": runtime.runtime_manifest_signature,
        "explainer_signature": explainer.signature, "question_bank_sha256": hashes["question_bank"],
        "question_bank_signature": question["source_bank_signature"],
        "question_bank_report_sha256": hashes["question_bank_report"],
        "tutorial_sha256": hashes["tutorial"], "tutorial_signature": tutorial_signature,
        "tutorial_scene_id": tutorial_scene["id"], "tutorial_scene_fingerprint": tutorial_scene["fingerprint"],
        "tutorial_successor_state_sha256": tutorial_scene["snapshot"]["r41_diagnostic_conflict"]["binding_sha256"],
        "tutorial_snapshot_sha256": digest(tutorial_scene["snapshot"]),
        "runtime_audit_sha256": digest(runtime_rows),
        "release_sources_sha256": package["release_sources_sha256"],
        "package_contract_sha256": digest(package)}
    if set(bindings) != BINDING_FIELDS:
        raise AssertionError("Diagnostic v8 admission binding schema differs")
    for name, value in bindings.items():
        if name != "tutorial_scene_id": _sha(value, name)
    return {"bindings": bindings,
        "artifacts": {name: {"path": relatives[name], "sha256": hashes[name]} for name in ARTIFACT_NAMES},
        "gates": {name: True for name in GATE_NAMES}, "sources": sources,
        "package_contract": package, "runtime_audit": runtime_rows}


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink() or path.parent.is_symlink() or path.parent.resolve() != path.parent:
        raise ValueError("Diagnostic admission output path is unsafe")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                         | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write((canonical(value) + "\n").encode("utf-8")); stream.flush(); os.fsync(stream.fileno())


def build_admission(paths: Mapping[str, str | Path], *, output: str | Path) -> dict[str, Any]:
    output_path = Path(output).expanduser()
    if not output_path.is_absolute(): output_path = ROOT / output_path
    output_path = output_path.absolute()
    try: relative = output_path.relative_to(ROOT.resolve()).as_posix()
    except ValueError: raise ValueError("Diagnostic admission output must stay inside repository") from None
    if output_path.exists() or output_path.is_symlink(): raise FileExistsError(output_path)
    checked = validate_components(paths)
    value = {"version": VERSION, "status": STATUS, "admitted": True,
        "namespace": NAMESPACE, "pilot_class": NAMESPACE,
        "behavior_performance_gate_passed": False, "behavior_performance_gate_waived": True,
        "waiver_scope": ["behavior_performance"], "formal_ready": False,
        "formal_sample_eligible": False, "human_explanation_effect_validated": False,
        "data_persistent": False, "runtime_action_override": False,
        "internal_diagnostic_only": True, "test_fixture": False,
        "bindings": checked["bindings"], "artifacts": checked["artifacts"],
        "gates": checked["gates"], "sources": checked["sources"],
        "package_contract": checked["package_contract"], "self_path": relative}
    _reject_sensitive(value, "diagnostic admission"); _write_new(output_path, value)
    try: read_saved_admission(output_path, expected_sha256=file_hash(output_path), components=paths)
    except BaseException: output_path.unlink(missing_ok=True); raise
    return value


def read_saved_admission(path: str | Path, *, expected_sha256: str,
        components: Mapping[str, str | Path]) -> dict[str, Any]:
    saved_path, relative = _artifact_path(path, "diagnostic admission")
    if file_hash(saved_path) != _sha(expected_sha256, "diagnostic admission"):
        raise ValueError("Diagnostic admission bytes differ")
    value = _strict_json(saved_path, "diagnostic admission")
    if (set(value) != TOP_LEVEL_FIELDS or value.get("version") != VERSION
            or value.get("status") != STATUS or value.get("admitted") is not True
            or value.get("namespace") != NAMESPACE or value.get("pilot_class") != NAMESPACE
            or value.get("behavior_performance_gate_passed") is not False
            or value.get("behavior_performance_gate_waived") is not True
            or value.get("waiver_scope") != ["behavior_performance"]
            or value.get("formal_ready") is not False or value.get("formal_sample_eligible") is not False
            or value.get("human_explanation_effect_validated") is not False
            or value.get("data_persistent") is not False or value.get("runtime_action_override") is not False
            or value.get("internal_diagnostic_only") is not True or value.get("test_fixture") is not False
            or value.get("self_path") != relative or not isinstance(value.get("bindings"), dict)
            or set(value["bindings"]) != BINDING_FIELDS
            or value.get("gates") != {name: True for name in GATE_NAMES}):
        raise ValueError("Exact non-formal diagnostic v8 admission required")
    checked = validate_components(components)
    for key in ("bindings", "artifacts", "gates", "sources", "package_contract"):
        if canonical(value.get(key)) != canonical(checked[key]):
            raise ValueError("Diagnostic admission " + key + " differs from live evidence")
    _reject_sensitive(value, "diagnostic admission"); return deepcopy(value)


__all__ = ["VERSION", "STATUS", "NAMESPACE", "FIXED_ACTOR_SHA256",
    "EXACT_ACTION_NAMES", "EXACT_CLASSES", "ARTIFACT_NAMES", "GATE_NAMES",
    "FINAL_ONCE_LEDGER_ARTIFACTS",
    "BINDING_FIELDS", "TOP_LEVEL_FIELDS", "MAX_PROGRAM_BYTES",
    "MAX_COMPONENT_ITERATIONS", "MAX_TOTAL_ITERATIONS", "MAX_TOTAL_TREES",
    "MAX_TOTAL_NODES", "MAXIMUM_TREE_DEPTH", "EXPECTED_RETIRED_HOLDOUTS",
    "validate_v8_program_identity",
    "source_closure", "package_contract", "validate_components",
    "build_admission", "read_saved_admission"]
