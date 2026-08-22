"""Deterministic conversational realization over the environment-neutral IR."""

from __future__ import annotations

import re
from typing import Any, Mapping

from .explanation_ir import (
    ExplanationDocumentV3,
    ExplanationIR,
    ExplanationIRCompiler,
    ExplanationLayer,
    ExplanationSemantics,
    render_ir,
    validate_document,
)
from .language_policy import normalize_supported_language
from .schemas import EvidenceBundle
from .tokenizer import StructuredTransformerBackend


class ExecutionGroundedExplanationGenerator:
    """Compile live evidence and render it without a realization-model call.

    ``backend`` remains an optional constructor argument so existing composition
    roots do not need a second migration.  It is deliberately never invoked:
    the only online generative call belongs to the QuestionIR parser.
    """

    def __init__(
        self,
        backend: StructuredTransformerBackend | None = None,
        *,
        semantics: ExplanationSemantics | None = None,
    ) -> None:
        self.backend = backend
        self.semantics = semantics
        self.compiler = ExplanationIRCompiler()
        self.last_grounding: dict[str, Any] = {}
        self.last_verified_grounding: dict[str, Any] = {}
        self.last_document: ExplanationDocumentV3 | None = None
        self.last_raw_document: ExplanationDocumentV3 | None = None
        self.last_raw_candidate_text: str | None = None
        self.last_raw_payload: Mapping[str, Any] | None = None
        self.last_ir: ExplanationIR | None = None
        self.last_model_call_count = 0
        self.last_generation_metrics: dict[str, Any] = {
            "input_tokens": 0,
            "output_tokens": 0,
        }

    def compile_ir(
        self,
        evidence: EvidenceBundle,
        *,
        include_program_trace: bool,
        language: str = "auto",
    ) -> ExplanationIR:
        return self.compiler.compile(
            evidence,
            include_program_trace=include_program_trace,
            requested_language=language,
            semantics=self.semantics,
        )

    def generate_document(
        self,
        evidence: EvidenceBundle,
        *,
        include_program_trace: bool,
        language: str = "auto",
        explanation_ir: ExplanationIR | None = None,
    ) -> ExplanationDocumentV3:
        self.last_verified_grounding = {}
        self.last_model_call_count = 0
        self.last_generation_metrics = {"input_tokens": 0, "output_tokens": 0}
        ir = explanation_ir or self.compile_ir(
            evidence,
            include_program_trace=include_program_trace,
            language=language,
        )
        self.last_ir = ir

        document = render_ir(ir)
        issues = validate_document(document, ir)
        if issues:
            # Keep the structural failure visible to callers and logs.  There
            # is no model repair and no technical prose fallback.
            document = render_ir(ir, issues=issues)

        self.last_document = document
        self.last_raw_document = document
        self.last_raw_candidate_text = document.text
        self.last_raw_payload = document.to_dict()
        self.last_grounding = self.grounding_for_document(
            ir,
            document,
            model_call_count=0,
        )
        return document

    def grounding_for_document(
        self,
        ir: ExplanationIR,
        document: ExplanationDocumentV3,
        *,
        model_call_count: int = 0,
    ) -> dict[str, Any]:
        program_ids = tuple(
            dict.fromkeys(
                evidence_id
                for unit in ir.trace_units
                if unit.unit_id in document.used_unit_ids
                for evidence_id in unit.evidence_ids
            )
        )
        used_evidence_ids = tuple(
            dict.fromkeys(
                evidence_id
                for unit in ir.units
                if unit.unit_id in document.used_unit_ids
                for evidence_id in unit.evidence_ids
            )
        )
        return {
            "schema_version": "explanation-grounding.v3",
            "explanation_mode": "rcpd_trace" if ir.trace_units else "no_trace",
            "explanation_ir_hash": ir.ir_hash,
            "explanation_ir": ir.to_dict(),
            "message_plan": document.message_plan.to_dict(),
            "explanation_document": document.to_dict(),
            "raw_candidate": {
                "text": document.text,
                "document": document.to_dict(),
                "used_unit_ids": document.used_unit_ids,
            },
            "semantic_plan": _semantic_plan_projection(ir),
            "used_unit_ids": document.used_unit_ids,
            "used_evidence_ids": used_evidence_ids,
            "covered_requirement_keys": document.used_unit_ids,
            "generation_recovery": {
                "mode": "deterministic_conversational",
                "validation_issues": document.validation_issues,
            },
            "model_call_count": int(model_call_count),
            "realization_model_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "display_program_evidence_ids": program_ids,
            "preauthored_explanation_text_used": False,
        }

    def generate(
        self,
        evidence: EvidenceBundle,
        *,
        include_program_trace: bool,
        language: str = "auto",
    ) -> str:
        return self.generate_document(
            evidence,
            include_program_trace=include_program_trace,
            language=language,
        ).text

    def render_verified(
        self,
        evidence: EvidenceBundle,
        *,
        language: str = "auto",
        include_program_trace: bool,
    ) -> str:
        ir = self.compile_ir(
            evidence,
            include_program_trace=include_program_trace,
            language=language,
        )
        document = render_ir(ir)
        self.last_verified_grounding = {
            "explanation_ir": ir.to_dict(),
            "explanation_ir_hash": ir.ir_hash,
            "explanation_document": document.to_dict(),
            "display_program_evidence_ids": tuple(
                evidence_id
                for unit in ir.trace_units
                for evidence_id in unit.evidence_ids
            ),
        }
        return document.text

    def _ensure_requested_output_language(
        self,
        answer: str,
        *,
        requested_language: str,
        question: str,
    ) -> str:
        """Compatibility helper; deterministic rendering never rewrites text."""

        desired = normalize_supported_language(requested_language, text=question)
        self.last_grounding["language_audit"] = {
            "requested": desired,
            "detected": desired,
            "rewritten": False,
        }
        return answer.strip()


def _semantic_plan_projection(ir: ExplanationIR) -> dict[str, Any]:
    """Read-only projection for v1/v2 logs and paired-analysis compatibility."""

    def records(layer: ExplanationLayer) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "evidence_id": unit.evidence_ids[0] if unit.evidence_ids else "",
                "requirement_key": (
                    f"{unit.value.get('phase')}.objective_context"
                    if unit.layer == ExplanationLayer.TASK_GOAL
                    and isinstance(unit.value, Mapping)
                    and unit.value.get("phase")
                    else unit.unit_id
                ),
                "kind": unit.predicate,
                "predicate": unit.predicate,
                "typed_value": unit.value,
                "phase": (
                    str(unit.value.get("phase", ""))
                    if isinstance(unit.value, Mapping)
                    else ""
                ),
                "action_code": (
                    str(unit.value.get("action", ""))
                    if isinstance(unit.value, Mapping)
                    else ""
                ),
                "reference_text": unit.reference_text,
                "trace_derived": unit.trace_derived,
            }
            for unit in ir.units
            if unit.layer == layer
        )

    mandatory = tuple(
        (
            f"{unit.value.get('phase')}.objective_context"
            if unit.layer == ExplanationLayer.TASK_GOAL
            and isinstance(unit.value, Mapping)
            and unit.value.get("phase")
            else unit.unit_id
        )
        for unit in ir.units
        if unit.mandatory
    )
    evidence_ids = tuple(
        dict.fromkeys(
            evidence_id
            for unit in ir.units
            for evidence_id in unit.evidence_ids
        )
    )
    return {
        "schema_version": "semantic-plan-compat.v3",
        "language_key": "zh" if ir.language == "zh-CN" else "en",
        "target": ir.target,
        "query_kind": ir.query_kind,
        "action_facts": (
            *records(ExplanationLayer.POLICY_PROPOSAL),
            *records(ExplanationLayer.FINAL_ACTION),
        ),
        "objective_facts": records(ExplanationLayer.TASK_GOAL),
        "reason_facts": (
            *records(ExplanationLayer.COORDINATION),
            *records(ExplanationLayer.ACTION_EFFECT),
        ),
        "program_facts": records(ExplanationLayer.PROPOSAL_RATIONALE),
        "intervention_facts": records(ExplanationLayer.COUNTERFACTUAL),
        "reason_scope": (
            "recorded_arbitration"
            if records(ExplanationLayer.COORDINATION)
            else "policy_output"
        ),
        "mandatory_requirement_keys": mandatory,
        "mandatory_evidence_ids": evidence_ids,
        "available_evidence_ids": evidence_ids,
    }


def _explanation_semantic_plan(
    evidence: EvidenceBundle,
    *,
    requested_language: str,
    include_program: bool,
) -> dict[str, Any]:
    """Compatibility view backed by the environment-neutral IR."""

    ir = ExplanationIRCompiler().compile(
        evidence,
        include_program_trace=include_program,
        requested_language=requested_language,
    )
    plan = _semantic_plan_projection(ir)
    plan["_explanation_ir"] = ir
    return plan


def _render_semantic_plan(plan: Mapping[str, Any]) -> str:
    ir = plan.get("_explanation_ir")
    if isinstance(ir, ExplanationIR):
        return render_ir(ir).text
    records = tuple(
        item
        for key in ("action_facts", "objective_facts", "program_facts", "reason_facts")
        for item in plan.get(key, ())
        if isinstance(item, Mapping)
    )
    return " ".join(
        str(item.get("reference_text", "")).strip()
        for item in records
        if str(item.get("reference_text", "")).strip()
    )


def _output_language_instruction(requested: str, *, question: str) -> str:
    language = normalize_supported_language(requested, text=question)
    return "Simplified Chinese only" if language == "zh-CN" else "English only"


def _plain_explanation_issues(answer: str) -> tuple[str, ...]:
    issues: list[str] = []
    if not answer.strip():
        issues.append("answer is empty")
    if "锟" in answer or re.search(r"\\x[0-9a-fA-F]{2}", answer):
        issues.append("answer contains invalid Unicode text")
    if re.search(r'"[A-Za-z_][A-Za-z0-9_]*"\s*:', answer):
        issues.append("answer contains a raw JSON field")
    if re.search(r"(?<![\w.])-?\d+(?:\.\d+)?[eE][+-]?\d+", answer):
        issues.append("answer contains scientific-notation evidence residue")
    if len(answer) > 1200:
        issues.append("answer is too long")
    return tuple(issues)


__all__ = [
    "ExecutionGroundedExplanationGenerator",
    "_explanation_semantic_plan",
    "_output_language_instruction",
    "_plain_explanation_issues",
    "_render_semantic_plan",
]
