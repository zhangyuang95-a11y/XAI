"""Strict saved-evidence assembly for the qualified alignment local pilot.

This reader performs no training, sampling, development selection, acceptance
collection, or fresh model inference.  It reconstructs the final Alignment
runtime from externally hashed files, delegates capability verification to the
original saved-episode reader, delegates explanation-system qualification and
the full saved answer audit to their completed-result readers, and loads the
independently replayed question bank.  Only this release layer may open the
local study; every component keeps its own study and release flags closed.
"""
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
import json
import os
import re

from backend import warehouse_alignment_runtime as runtime_api
from backend import warehouse_alignment_diverse_explanation as explanation_api
from backend.training import warehouse_family_alignment_evaluation as capability_api
from backend.training import warehouse_family_alignment_run as alignment_run
from backend.training import warehouse_family_explanation_system_run as system_api
from backend.training import warehouse_alignment_diverse_answer_run as answer_api
from backend.training import warehouse_native_cycle_budget as budget_api
from backend.training.warehouse_native_common import ROOT, canonical, digest, file_hash
from ui import warehouse_alignment_server as server_api
from ui.warehouse_alignment_metadata_bank_view import FrozenAlignmentMetadataBank
from ui.warehouse_native_release import analysis_protocol


VERSION = "warehouse-alignment-local-pilot-release.v1"
BUNDLE_VERSION = "warehouse-alignment-local-pilot-evidence-bundle.v1"
_BUNDLE_FIELDS = frozenset((
    "version", "test_fixture", "actor", "protocol", "scenarios", "program",
    "system_acceptance", "answer_audit", "capability", "question_bank",
))
_CAPABILITY_FIELDS = frozenset((
    "run_directory", "prepared_sha256", "budget_sha256", "terminal_sha256",
    "boundary_sha256", "validation_manifest_sha256", "report_sha256",
    "reference_report", "random_report",
))
_BANK_FIELDS = frozenset((
    "directory", "prepared_sha256", "replay_receipt_sha256",
    "pool_manifest_sha256",
))
_SYSTEM_FIELDS = frozenset(("directory", "report_sha256", "plan_sha256"))
_ANSWER_FIELDS = frozenset(("directory", "receipt_sha256"))
_ANSWER_RESULT_FIELDS = frozenset((
    "receipt", "generated_report", "verification_report", "passed",
    "reader_environment_steps", "reader_NN_queries", "release_ready",
))
_ANSWER_RECEIPT_FIELDS = frozenset((
    "version", "status", "passed", "plan_file_sha256", "reports",
    "bindings", "sources_sha256", "full_matrix", "phase_caps",
    "research_qualification_evaluated", "release_ready",
))
_ANSWER_REPORT_FIELDS = frozenset((
    "version", "operation", "cases", "trajectories", "passed", "coverage",
    "required_coverage_complete", "actual_tree_disagreement_cases",
    "mismatch_disclosure_cases", "execution", "independently_verified",
    "research_qualification_evaluated", "release_ready", "bindings",
    "sources", "test_fixture", "input_report_sha256",
))
_MANIFEST_FIELDS = frozenset((
    "version", "status", "namespace", "test_fixture", "formal_ready",
    "online_deployment", "human_explanation_effect_validated",
    "bundle_file_sha256", "bundle_sha256", "evidence", "sources", "analysis",
    "model_ready", "explanation_ready", "study_ready", "participant_enabled",
    "explanation_qualified", "release_ready", "web_integration_completed",
))


def _sha(value, name="SHA256"):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Explicit lowercase SHA256 required: " + name)
    return value


def _same(left, right, reason):
    if canonical(left) != canonical(right):
        raise ValueError(reason)


def _pairs(values):
    result = {}
    for name, value in values:
        if name in result:
            raise ValueError("Duplicate JSON field: " + name)
        result[name] = value
    return result


def _parse(raw):
    def invalid(value):
        raise ValueError("Non-finite JSON value: " + value)
    value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=invalid)
    if type(value) is not dict:
        raise ValueError("Top-level evidence JSON object required")
    return value


def _directory(path):
    value = Path(path).expanduser().absolute()
    if value.resolve() != value or value.is_symlink() or not value.is_dir():
        raise ValueError("Existing canonical non-symlink directory required")
    return value


def _read(path, expected, label):
    _sha(expected, label)
    value = Path(path).expanduser().absolute()
    if value.resolve() != value or value.is_symlink() or not value.is_file():
        raise ValueError("Regular canonical evidence file required: " + label)
    raw = value.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError("Externally bound bytes differ: " + label)
    return raw


def _json(path, expected, label):
    return _parse(_read(path, expected, label))


def _file_item(item, label):
    if type(item) is not dict or set(item) != {"path", "sha256"}:
        raise ValueError("Exact path/SHA binding required: " + label)
    return Path(item["path"]).expanduser().absolute(), _read(
        item["path"], item["sha256"], label
    )


def _bound_in(root, binding, label):
    if type(binding) is not dict or set(binding) != {"path", "sha256", "size"}:
        raise ValueError("Saved run artifact binding differs: " + label)
    relative = Path(binding["path"])
    if (relative.is_absolute() or ".." in relative.parts
            or relative.as_posix() != binding["path"]):
        raise ValueError("Saved run artifact escapes its root: " + label)
    path = (root / relative).absolute()
    raw = _read(path, binding["sha256"], label)
    if type(binding["size"]) is not int or binding["size"] != len(raw):
        raise ValueError("Saved run artifact size differs: " + label)
    return path, raw


def _merge_sources(target, values):
    if type(values) is not dict or not values:
        raise ValueError("Complete source closure required")
    for name, value in values.items():
        _sha(value, "source " + str(name))
        path = Path(name)
        if (type(name) is not str or path.is_absolute() or ".." in path.parts
                or path.as_posix() != name):
            raise ValueError("Unsafe source path")
        if name in target and target[name] != value:
            raise ValueError("Release component source closures disagree: " + name)
        target[name] = value


def _verify_sources(values):
    for name, expected in values.items():
        path = ROOT / name
        if path.resolve() != path or file_hash(path) != expected:
            raise ValueError("Release source changed: " + name)
    return deepcopy(values)


def release_sources(context=None):
    """Return the complete code closure without reading any experiment pool."""
    values = {}
    for records in (
        runtime_api.runtime_sources(),
        explanation_api.explanation_sources(),
        system_api.execution_sources(),
        answer_api.sources(),
        capability_api.execution_sources(),
        server_api.service_sources(),
    ):
        _merge_sources(values, records)
    for module in (alignment_run, budget_api):
        path = Path(module.__file__)
        _merge_sources(values, {str(path.relative_to(ROOT)): file_hash(path)})
    for path in (Path(__file__), ROOT / "ui/warehouse_alignment_metadata_bank_view.py"):
        _merge_sources(values, {str(path.relative_to(ROOT)): file_hash(path)})
    if context is not None:
        for records in (
            context.runtime.sources,
            context.explainer.sources,
            context.question_bank.sources,
        ):
            _merge_sources(values, records)
    return _verify_sources(dict(sorted(values.items())))


@dataclass
class _Context:
    runtime: object
    scenarios: dict
    explainer: object
    question_bank: object
    evidence: dict
    source_binding: dict
    manifest_sha256: str = ""
    signature: str = ""
    release: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    closed: bool = False

    def close(self):
        self.closed = True


def _qualified_payload(value):
    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()
    if type(value) is not dict:
        raise ValueError("System acceptance must return the qualified executable program")
    return value


def _system(item, program_payload, expected_program_sha256):
    if type(item) is not dict or set(item) != _SYSTEM_FIELDS:
        raise ValueError("Exact system-acceptance anchors required")
    root = _directory(item["directory"])
    _read(root / "report.json", item["report_sha256"], "system report")
    _read(root / "plan.json", item["plan_sha256"], "system plan")
    result = system_api.read_result(
        root,
        expected_report_sha256=item["report_sha256"],
        expected_plan_sha256=item["plan_sha256"],
    )
    result_fields = {
        "report", "qualified_program", "receipt", "system_qualified",
        "free_question_answer_qualified", "release_ready",
        "reader_environment_steps", "reader_NN_queries", "reader_PT_loads",
    }
    if (type(result) is not dict or set(result) != result_fields
            or result.get("system_qualified") is not True
            or result.get("free_question_answer_qualified") is not False
            or result.get("release_ready") is not False
            or any(result.get(name) != 0 for name in (
                "reader_environment_steps", "reader_NN_queries", "reader_PT_loads"
            ))
            or type(result.get("report")) is not dict
            or type(result.get("receipt")) is not dict):
        raise ValueError("Completed system-level explanation qualification is required")
    qualified = _qualified_payload(result.get("qualified_program"))
    _same(qualified, program_payload,
          "Release program differs from the system-qualified executable program")
    # The strict tree-pair result remains visible as a diagnostic.  It is never
    # promoted to a gate by this adapter.
    acceptance = result["report"].get("record_acceptance", result["report"])
    qualified_metadata = qualified.get("metadata", {}).get(
        "explanation_system_acceptance", {}
    )
    if (type(acceptance) is not dict
            or acceptance.get("records_passed") is not True
            or acceptance.get("tree_pair_diagnostic_is_gate") is not False
            or result["receipt"].get("version") != system_api.VERSION
            or result["receipt"].get("report_file_sha256") != item["report_sha256"]
            or result["receipt"].get("plan_file_sha256") != item["plan_sha256"]
            or result["receipt"].get("qualified_program_file_sha256")
                != _sha(expected_program_sha256, "system-qualified program")
            or result["receipt"].get("ordinary_tree_source_program_sha256")
                != qualified_metadata.get("ordinary_tree_source_program_sha256")
            or result["receipt"].get("qualification_scope")
                != "ordinary_tree_and_isolated_nn_engine"
            or result["receipt"].get("free_question_answer_qualified") is not False
            or result["receipt"].get("release_ready") is not False):
        raise ValueError("System acceptance lost the non-gating strict tree diagnostic")
    return result, {
        "directory": str(root),
        "report_sha256": item["report_sha256"],
        "plan_sha256": item["plan_sha256"],
        "result_sha256": digest(result),
        "receipt_sha256": digest(result["receipt"]),
        "system_qualified": True,
        "tree_pair_diagnostic": deepcopy(acceptance.get("tree_pair_diagnostic")),
        "tree_pair_diagnostic_is_gate": False,
    }


def _answer_execution(report, operation):
    """Return the durable ACK count after checking the saved phase summary."""
    execution = report.get("execution")
    counts = execution.get("actual_execution") if type(execution) is dict else None
    if type(counts) is not dict:
        raise ValueError("Answer audit lacks physical execution accounting")
    acknowledged = counts.get("acknowledged_steps")
    phases = counts.get("steps_by_phase")
    required_phases = {"fixed_prefix", "renderer"}
    if operation == "verify":
        required_phases.add("independent_oracle")
    valid_phases = (type(phases) is dict and bool(phases)
                    and all(type(value) is int and value > 0
                            for value in phases.values()))
    if (execution.get("auxiliary_step_budget") != answer_api.PHASE_CAP
            or execution.get("pending_operation") is not None
            or execution.get("automatic_retry") is not False
            or execution.get("accounting_complete") is not True
            or type(acknowledged) is not int or acknowledged <= 0
            or counts.get("environment_steps") != acknowledged
            or counts.get("environment_step_attempts") != acknowledged
            or not valid_phases
            or sum(phases.values()) != acknowledged
            or not required_phases.issubset(phases)
            or type(counts.get("numpy_forward_calls")) is not int
            or counts["numpy_forward_calls"] <= 0
            or type(counts.get("numpy_actor_load_attempts")) is not int
            or counts["numpy_actor_load_attempts"] != 1
            or type(counts.get("numpy_actor_loads")) is not int
            or counts["numpy_actor_loads"] != 1
            or counts.get("neural_updates") != 0
            or counts.get("torch_loads") != 0):
        raise ValueError("Answer audit phase is not fully ACK-bound actual execution")
    return acknowledged


def _answer_audit(item, runtime, program_path, scenarios):
    """Admit the immutable 480-answer receipt without running NN or physics."""
    if type(item) is not dict or set(item) != _ANSWER_FIELDS:
        raise ValueError("Exact answer-audit anchors required")
    root = _directory(item["directory"])
    receipt_sha256 = _sha(item["receipt_sha256"], "answer audit receipt")
    saved_receipt = _json(
        root / "answer_receipt.json", receipt_sha256, "answer audit receipt"
    )
    expected_bindings = answer_api.bindings(runtime, program_path, scenarios)
    result = answer_api.read_receipt(
        root,
        expected_receipt_sha256=receipt_sha256,
        expected_bindings=expected_bindings,
    )
    if (type(result) is not dict or set(result) != _ANSWER_RESULT_FIELDS
            or result.get("passed") is not True
            or result.get("reader_environment_steps") != 0
            or result.get("reader_NN_queries") != 0
            or result.get("release_ready") is not False):
        raise ValueError("Completed pure-reader answer audit is required")
    receipt = result.get("receipt")
    generated = result.get("generated_report")
    verified = result.get("verification_report")
    if (type(receipt) is not dict or set(receipt) != _ANSWER_RECEIPT_FIELDS
            or type(generated) is not dict or set(generated) != _ANSWER_REPORT_FIELDS
            or type(verified) is not dict or set(verified) != _ANSWER_REPORT_FIELDS):
        raise ValueError("Answer audit receipt/report schema differs")
    _same(saved_receipt, receipt, "Answer reader returned another receipt")
    answer_sources = answer_api.sources()
    phases = {"generate", "verify"}
    if (receipt.get("version") != answer_api.VERSION
            or receipt.get("status") != "answers_verified"
            or receipt.get("passed") is not True
            or receipt.get("full_matrix") != 480
            or receipt.get("bindings") != expected_bindings
            or receipt.get("sources_sha256") != digest(answer_sources)
            or receipt.get("phase_caps") != dict.fromkeys(
                answer_api.journal.PHASES, answer_api.PHASE_CAP
            )
            or type(receipt.get("reports")) is not dict
            or set(receipt["reports"]) != phases
            or receipt.get("research_qualification_evaluated") is not False
            or receipt.get("release_ready") is not False):
        raise ValueError("Answer receipt does not bind the full verified matrix")
    _sha(receipt.get("plan_file_sha256"), "answer audit plan")

    summaries = {}
    for operation, report in (("generate", generated), ("verify", verified)):
        cases = report.get("cases")
        case_ids = ([row.get("case_id") for row in cases]
                    if type(cases) is list and all(type(row) is dict for row in cases)
                    else [])
        if (report.get("version") != answer_api.VERSION
                or report.get("operation") != operation
                or report.get("test_fixture") is not False
                or report.get("bindings") != expected_bindings
                or canonical(report.get("sources")) != canonical(answer_sources)
                or len(case_ids) != 480 or len(set(case_ids)) != 480
                or any(type(case_id) is not str or not case_id for case_id in case_ids)
                or report.get("research_qualification_evaluated") is not False
                or report.get("release_ready") is not False):
            raise ValueError("Answer audit does not contain the exact real 480-case phase")
        acknowledged = _answer_execution(report, operation)
        summaries[operation] = {
            "report_sha256": digest(report),
            "cases": len(case_ids),
            "acknowledged_steps": acknowledged,
            "steps_by_phase": deepcopy(report["execution"]["actual_execution"]["steps_by_phase"]),
            "independently_verified": report["independently_verified"],
            "passed": report["passed"],
        }
    if (generated.get("passed") is not False
            or generated.get("independently_verified") is not False
            or generated.get("required_coverage_complete") is not False
            or generated.get("input_report_sha256") is not None
            or verified.get("passed") is not True
            or verified.get("independently_verified") is not True
            or verified.get("required_coverage_complete") is not True
            or verified.get("input_report_sha256") != digest(generated)
            or digest(generated.get("trajectories"))
                != digest(verified.get("trajectories"))
            or any(row.get("passed") is not True for row in verified["cases"])):
        raise ValueError("Independent generate/verify answer outcome did not pass")
    # Detect a receipt swap after the delegated reader and all semantic checks.
    _read(root / "answer_receipt.json", receipt_sha256, "answer audit receipt")
    evidence = {
        "version": answer_api.VERSION,
        "directory": str(root),
        "receipt_sha256": receipt_sha256,
        "receipt_record_sha256": digest(receipt),
        "receipt": deepcopy(receipt),
        "bindings": deepcopy(expected_bindings),
        "generate": summaries["generate"],
        "verify": summaries["verify"],
        "result_sha256": digest(result),
        "full_matrix": 480,
        "required_coverage_complete": True,
        "passed": True,
        "reader_environment_steps": 0,
        "reader_NN_queries": 0,
        "release_ready": False,
    }
    return result, evidence


def _capability(item, runtime, actor_path, protocol, scenarios):
    if type(item) is not dict or set(item) != _CAPABILITY_FIELDS:
        raise ValueError("Complete original capability evidence anchors required")
    run = _directory(item["run_directory"])
    prepared = _json(run / "prepared.json", item["prepared_sha256"], "capability prepared")
    if (prepared.get("version") != alignment_run.VERSION
            or prepared.get("qualification_granted") is not False
            or prepared.get("identity", {}).get("test_fixture") is not False):
        raise ValueError("Genuine frozen alignment run preparation required")
    if file_hash(run / budget_api.FILENAME) != _sha(item["budget_sha256"], "capability budget"):
        raise ValueError("Externally bound alignment cycle budget differs")
    ledger = budget_api.CycleBudget(run, prepared["identity"]).read()
    if any(ledger["branches"][branch][kind]["pending"] is not None
           for branch in ledger["branches"] for kind in ("ppo", "evaluation")):
        raise ValueError("Capability run has unresolved work")
    terminal = _json(run / "terminal.json", item["terminal_sha256"], "capability terminal")
    endpoint = alignment_run.PPO_CAP
    if (terminal.get("version") != alignment_run.VERSION
            or terminal.get("status") != "fixed_50k_pair_completed"
            or terminal.get("endpoint") != endpoint
            or terminal.get("capability_gates", {}).get("feedback") is not True
            or terminal.get("explanation_qualified") is not False
            or terminal.get("whole_goal_completed") is not False):
        raise ValueError("Completed final alignment capability boundary required")
    boundary_path = run / "branches" / "feedback" / "boundaries" / f"step_{endpoint:07d}.json"
    boundary = _json(boundary_path, item["boundary_sha256"], "capability boundary")
    if (boundary.get("version") != alignment_run.VERSION
            or boundary.get("branch") != "feedback" or boundary.get("step") != endpoint
            or boundary.get("capability_passed") is not True):
        raise ValueError("Final feedback Actor did not pass the original capability gate")
    saved_actor_path, _ = _bound_in(run, boundary.get("actor"), "capability Actor")
    report_path, _ = _bound_in(run, boundary.get("report"), "capability report")
    if (saved_actor_path != actor_path or boundary["actor"]["sha256"] != runtime.actor_sha256
            or boundary["report"]["sha256"] != item["report_sha256"]):
        raise ValueError("Capability evidence belongs to another final Actor or report")
    _read(report_path.parent / "manifest.json", item["validation_manifest_sha256"],
          "capability episode manifest")

    protocols_path, protocols_raw = _bound_in(
        run, prepared.get("protocols"), "alignment run protocols"
    )
    del protocols_path
    protocols = _parse(protocols_raw)
    _same(protocols.get("feedback"), protocol,
          "Capability run used another final Actor protocol")
    reference_path, reference_raw = _file_item(item["reference_report"], "reference report")
    random_path, random_raw = _file_item(item["random_report"], "random report")
    del reference_path, random_path
    reference, random = _parse(reference_raw), _parse(random_raw)
    confirmed = {
        opid for opid, operation in ledger["operations"].items()
        if operation["status"] == "acknowledged"
        and operation["request"]["kind"] == "evaluation"
        and operation["request"]["branch"] == "feedback"
    }
    report = capability_api.read_existing(
        actor_path,
        scenarios["splits"]["validation"],
        protocol,
        report_path.parent,
        expected_actor_sha256=runtime.actor_sha256,
        reference_report=reference,
        random_report=random,
        confirmed_operation_ids=confirmed,
        expected_report_sha256=item["report_sha256"],
    )
    if (report.get("version") != capability_api.VERSION
            or report.get("status") != "completed"
            or report.get("test_fixture") is not False
            or report.get("strict_capability_eligible") is not True
            or report.get("actor_bindings", {}).get("actor_sha256") != runtime.actor_sha256):
        raise ValueError("Recomputed original capability gate did not pass")
    # Detect concurrent changes to every caller anchor not already re-read by
    # the delegated verifier.
    for path, expected, label in (
        (run / "prepared.json", item["prepared_sha256"], "capability prepared"),
        (run / budget_api.FILENAME, item["budget_sha256"], "capability budget"),
        (run / "terminal.json", item["terminal_sha256"], "capability terminal"),
        (boundary_path, item["boundary_sha256"], "capability boundary"),
        (report_path.parent / "manifest.json", item["validation_manifest_sha256"],
         "capability episode manifest"),
        (report_path, item["report_sha256"], "capability report"),
    ):
        _read(path, expected, label)
    return {
        "version": capability_api.VERSION,
        "run_directory": str(run),
        "actor_sha256": runtime.actor_sha256,
        "runtime_signature": runtime.signature,
        "protocol_sha256": runtime.protocol_sha256,
        "scenario_manifest_sha256": digest(scenarios),
        "report_sha256": item["report_sha256"],
        "report_semantic_sha256": digest(report),
        "confirmed_feedback_evaluations": len(confirmed),
        "strict_capability_eligible": True,
        "teacher_gate_diagnostic": terminal.get("teacher_gate"),
    }


def _bank(item, runtime, scenarios):
    if type(item) is not dict or set(item) != _BANK_FIELDS:
        raise ValueError("Complete frozen question-bank anchors required")
    bank = FrozenAlignmentMetadataBank(
        item["directory"],
        expected_prepared_sha256=item["prepared_sha256"],
        expected_replay_receipt_sha256=item["replay_receipt_sha256"],
        expected_runtime_signature=runtime.signature,
        expected_actor_sha256=runtime.actor_sha256,
        expected_protocol_sha256=runtime.protocol_sha256,
        expected_scenario_manifest_sha256=digest(scenarios),
        expected_pool_manifest_sha256=item["pool_manifest_sha256"],
        allow_test_fixture=False,
    )
    if (bank.content_eligible is not True or bank.test_fixture is not False
            or any(getattr(bank, name) is not False for name in
                   ("eligible", "participant_enabled", "formal_ready", "release_ready"))):
        raise ValueError("Independently replayed question bank is incomplete")
    bank.verify_binding()
    return bank


def _bundle(value):
    if (type(value) is not dict or set(value) != _BUNDLE_FIELDS
            or value.get("version") != BUNDLE_VERSION
            or value.get("test_fixture") is not False):
        raise ValueError("Exact genuine alignment release bundle required")
    for name in ("actor", "protocol", "scenarios", "program"):
        if type(value.get(name)) is not dict or set(value[name]) != {"path", "sha256"}:
            raise ValueError("Missing external artifact anchor: " + name)
        _sha(value[name]["sha256"], name)
    return value


def _material(bundle):
    bundle = _bundle(bundle)
    actor_path, _ = _file_item(bundle["actor"], "final Actor")
    protocol_path, protocol_raw = _file_item(bundle["protocol"], "final protocol")
    scenarios_path, scenarios_raw = _file_item(bundle["scenarios"], "original scenarios")
    program_path, program_raw = _file_item(bundle["program"], "qualified program")
    del protocol_path, scenarios_path
    protocol, scenarios, program_payload = (
        _parse(protocol_raw), _parse(scenarios_raw), _parse(program_raw)
    )
    runtime = runtime_api.AlignmentRuntime(
        actor_path,
        protocol=protocol,
        expected_actor_sha256=bundle["actor"]["sha256"],
        expected_protocol_sha256=digest(protocol),
        allow_test_fixture=False,
    )
    identity = runtime_api.verify(runtime, allow_test_fixture=False,
                                  expected_family=runtime_api.FAMILY)
    if (type(scenarios.get("splits")) is not dict
            or type(scenarios["splits"].get("play")) is not list
            or len(scenarios["splits"]["play"]) < 7
            or type(scenarios["splits"].get("validation")) is not list
            or len(scenarios["splits"]["validation"]) != 50
            or scenarios.get("test_fixture") is True
            or digest(scenarios) != runtime.actor.metadata.get("scenario_manifest_sha256")):
        raise ValueError("Original non-fixture study and validation scenarios differ")
    system_result, system_evidence = _system(
        bundle["system_acceptance"], program_payload, bundle["program"]["sha256"]
    )
    explainer = explanation_api.DiverseAlignmentExplainer(
        program_path,
        expected_program_sha256=bundle["program"]["sha256"],
        runtime=runtime,
        allow_test_fixture=False,
    )
    answer_result, answer_evidence = _answer_audit(
        bundle["answer_audit"], runtime, program_path, scenarios
    )
    capability = _capability(bundle["capability"], runtime, actor_path, protocol, scenarios)
    bank = _bank(bundle["question_bank"], runtime, scenarios)
    context = _Context(runtime, deepcopy(scenarios), explainer, bank, {}, {})
    try:
        service_identity, bank_identity, service_sources = server_api._components(
            runtime, explainer, scenarios,
            expected_scenarios=digest(scenarios), bank=bank, required_bank=True,
        )
        _same(service_identity, identity,
              "Release and production service runtime admission differ")
        sources = release_sources(context)
        evidence = {
            "version": BUNDLE_VERSION,
            "runtime_identity": identity,
            "actor_file_sha256": bundle["actor"]["sha256"],
            "protocol_file_sha256": bundle["protocol"]["sha256"],
            "scenario_file_sha256": bundle["scenarios"]["sha256"],
            "scenario_manifest_sha256": digest(scenarios),
            "program_file_sha256": bundle["program"]["sha256"],
            "program_content_sha256": explainer.program_content_sha256,
            "explainer_signature": explainer.signature,
            "system_acceptance": system_evidence,
            "system_result_sha256": digest(system_result),
            "answer_audit": answer_evidence,
            "answer_audit_result_sha256": digest(answer_result),
            "capability": capability,
            "question_bank_signature": bank.signature,
            "question_bank_summary_sha256": digest(bank.summary()),
            "question_bank_public_items_sha256": digest(bank.public_items()),
            "production_service_bank_identity": bank_identity,
            "production_service_sources_sha256": digest(service_sources),
            "all_component_gates_passed": True,
        }
        context.evidence = deepcopy(evidence)
        context.source_binding = deepcopy(sources)
        return context, evidence
    except BaseException:
        context.close()
        raise


def _write(path, raw):
    path = Path(path)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def assemble_local_release(bundle_path, *, expected_bundle_sha256, output):
    """Validate completed production evidence and write a new immutable release."""
    bundle_path = Path(bundle_path).expanduser().absolute()
    raw = _read(bundle_path, expected_bundle_sha256, "release bundle")
    bundle = _bundle(_parse(raw))
    context, evidence = _material(bundle)
    try:
        sources = release_sources(context)
        _same(sources, context.source_binding, "Release sources changed after admission")
        manifest = {
            "version": VERSION,
            "status": "local_pilot_technically_verified",
            "namespace": "local_pilot",
            "test_fixture": False,
            "formal_ready": False,
            "online_deployment": False,
            "human_explanation_effect_validated": False,
            "bundle_file_sha256": expected_bundle_sha256,
            "bundle_sha256": digest(bundle),
            "evidence": evidence,
            "sources": sources,
            "analysis": analysis_protocol(),
            "model_ready": True,
            "explanation_ready": True,
            "study_ready": True,
            "participant_enabled": True,
            "explanation_qualified": True,
            "release_ready": True,
            "web_integration_completed": True,
        }
        target = Path(output).expanduser().absolute()
        if (target.resolve() != target or target.exists() or target.parent.resolve() != target.parent
                or not target.parent.is_dir()):
            raise ValueError("Use a new canonical release directory")
        for item in (
            bundle["system_acceptance"], bundle["answer_audit"],
            bundle["question_bank"],
        ):
            source = _directory(item["directory"])
            if target == source or target.is_relative_to(source) or source.is_relative_to(target):
                raise ValueError("Release directory must be separate from evidence")
        target.mkdir()
        _write(target / "bundle.json", raw)
        _write(target / "manifest.json", (canonical(manifest) + "\n").encode())
        return {
            "version": VERSION,
            "status": manifest["status"],
            "manifest": str(target / "manifest.json"),
            "manifest_sha256": file_hash(target / "manifest.json"),
            "bundle_sha256": expected_bundle_sha256,
            "study_ready": True,
            "formal_ready": False,
        }
    finally:
        context.close()


def load_alignment_release(root, *, expected_manifest_sha256):
    """Rebuild a local-pilot context from only hash-bound saved evidence."""
    root = _directory(root)
    manifest = _json(root / "manifest.json", expected_manifest_sha256,
                     "alignment release manifest")
    if (set(manifest) != _MANIFEST_FIELDS or manifest.get("version") != VERSION
            or manifest.get("status") != "local_pilot_technically_verified"
            or manifest.get("namespace") != "local_pilot"
            or manifest.get("test_fixture") is not False
            or manifest.get("formal_ready") is not False
            or manifest.get("online_deployment") is not False
            or manifest.get("human_explanation_effect_validated") is not False
            or any(manifest.get(name) is not True for name in (
                "model_ready", "explanation_ready", "study_ready",
                "participant_enabled", "explanation_qualified", "release_ready",
                "web_integration_completed",
            ))):
        raise ValueError("Completed genuine alignment local-pilot manifest required")
    raw = _read(root / "bundle.json", manifest.get("bundle_file_sha256"),
                "stored release bundle")
    bundle = _bundle(_parse(raw))
    if digest(bundle) != _sha(manifest.get("bundle_sha256"), "bundle semantic hash"):
        raise ValueError("Stored release bundle semantics changed")
    context, evidence = _material(bundle)
    try:
        _same(evidence, manifest["evidence"], "Release component evidence changed")
        _same(manifest["analysis"], analysis_protocol(), "Frozen A/B analysis changed")
        _same(manifest["sources"], release_sources(context),
              "Complete release source closure changed")
        _same(manifest["sources"], context.source_binding,
              "Release context source binding changed")
        context.manifest_sha256 = expected_manifest_sha256
        context.signature = digest({
            "version": VERSION,
            "manifest_sha256": expected_manifest_sha256,
            "runtime_signature": context.runtime.signature,
            "program_sha256": context.explainer.program_sha256,
            "bank_signature": context.question_bank.signature,
            "scenario_manifest_sha256": digest(context.scenarios),
            "evidence_sha256": digest(evidence),
            "sources": context.source_binding,
        })
        context.release = {
            "status": manifest["status"],
            "namespace": "local_pilot",
            "model_ready": True,
            "explanation_ready": True,
            "study_ready": True,
            "participant_enabled": True,
            "explanation_qualified": True,
            "release_ready": True,
            "web_integration_completed": True,
            "formal_ready": False,
            "test_fixture": False,
            "qualification_evaluated": True,
            "message": {
                "zh": "本地预实验：模型、解释系统与问卷证据已通过技术验收，正式研究结论仍待人类实验。",
                "en": "Local pilot: model, explanation system, and questionnaire evidence passed technical acceptance; formal study conclusions remain pending.",
            },
        }
        context.provenance = {
            "version": VERSION,
            "namespace": "local_pilot",
            "manifest_sha256": expected_manifest_sha256,
            "actor_sha256": context.runtime.actor_sha256,
            "runtime_signature": context.runtime.signature,
            "protocol_sha256": context.runtime.protocol_sha256,
            "scenario_manifest_sha256": digest(context.scenarios),
            "program_sha256": context.explainer.program_sha256,
            "question_bank_signature": context.question_bank.signature,
            "system_acceptance_result_sha256": evidence["system_result_sha256"],
            "answer_audit_receipt_sha256": evidence["answer_audit"]["receipt_sha256"],
            "answer_audit_result_sha256": evidence["answer_audit_result_sha256"],
            "capability_report_sha256": evidence["capability"]["report_sha256"],
            "analysis": deepcopy(manifest["analysis"]),
            "source_binding": deepcopy(context.source_binding),
            "release": deepcopy(context.release),
        }
        return context
    except BaseException:
        context.close()
        raise


__all__ = [
    "VERSION", "BUNDLE_VERSION", "assemble_local_release",
    "load_alignment_release", "release_sources",
]
