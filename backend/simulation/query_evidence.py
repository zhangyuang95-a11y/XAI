"""Pure evidence, cache-key and counterfactual analysis helpers."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from backend.adapters.base import (
    EnvironmentAdapter,
    EnvironmentSnapshot,
    PolicyProtocol,
    RolloutFrame,
)
from core.program import ExecutableProgram
from backend.nlp.explanation_ir import (
    ExplanationDocumentV3,
    ExplanationIR,
)
from backend.nlp.schemas import (
    AtomicClaim,
    ClaimVerdict,
    ClaimVerdictStatus,
    EvidenceBundle,
    EvidenceRecord,
    QueryPlan,
)
from backend.simulation.counterfactual import CounterfactualResult
from backend.simulation.intervention import SceneEditResult


def _document_cache_key(
    ir_hash: str,
    *,
    policy_hash: str | None,
    program_hash: str | None,
    environment_id: str,
) -> str:
    payload = json.dumps(
        {
            "ir_hash": ir_hash,
            "policy_hash": policy_hash,
            "program_hash": program_hash,
            "environment_id": environment_id,
            "renderer": "conversational_ir_renderer",
            "schema_version": "explanation-document.v3",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _common_ir_cache_key(
    evidence: EvidenceBundle,
    *,
    language: str,
    policy_hash: str | None,
    program_hash: str | None,
    environment_id: str,
) -> str:
    target = str(
        evidence.policy_results.get(
            "target",
            evidence.direct_result.get("target", ""),
        )
    )
    payload = json.dumps(
        {
            "intent": evidence.query_plan.intent.value,
            "target_variables": tuple(evidence.query_plan.target_variables),
            # Study questions about the same action may request different
            # grounded dimensions (energy, teammate coordination, or shared
            # task allocation).  They must never share a cached common IR.
            "evidence_requirements": tuple(
                evidence.query_plan.evidence_requirements
            ),
            "desired_outcomes": dict(evidence.query_plan.desired_outcomes),
            "counterfactual": bool(
                evidence.query_plan.requires_scene_edit
            ),
            "show_position": _plan_requests_position(
                evidence.query_plan
            ),
            "target": target,
            "proposed_action": evidence.policy_results.get(
                "proposed_action",
                evidence.direct_result.get("proposed_action"),
            ),
            "executed_action": evidence.policy_results.get(
                "executed_action",
                evidence.direct_result.get("executed_action"),
            ),
            "action_resolution": evidence.policy_results.get(
                "action_resolution",
                evidence.direct_result.get("action_resolution", {}),
            ),
            "shared_task_context": evidence.policy_results.get(
                "shared_task_context",
                evidence.direct_result.get("shared_task_context", {}),
            ),
            "state_facts": evidence.state_facts,
            "interventions": evidence.interventions,
            "counterfactual_contrast": evidence.direct_result.get(
                "counterfactual_contrast",
                evidence.policy_results.get(
                    "counterfactual_contrast",
                    {},
                ),
            ),
            "language": language,
            "policy_hash": policy_hash,
            "program_hash": program_hash,
            "environment_id": environment_id,
            "schema_version": "explanation-ir.v2",
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _answer_cache_key(
    plan: QueryPlan,
    snapshot: EnvironmentSnapshot,
    *,
    condition: str,
    language: str,
    seed: int,
    policy_hash: str | None,
    program_hash: str | None,
    model_id: str,
) -> str:
    payload = {
        "schema_version": "query-answer.v3",
        "plan": plan.to_dict(),
        "snapshot": {
            "environment": snapshot.environment,
            "frame": snapshot.frame,
            "state": _cache_value(snapshot.state),
            "proposed_actions": dict(snapshot.proposed_actions),
            "executed_actions": dict(snapshot.executed_actions),
            "decision_metadata": {
                key: _cache_value(snapshot.metadata.get(key))
                for key in (
                    "action_resolution",
                    "decision_outcome_frame",
                    "decision_evidence_aligned",
                    "environment_events",
                    "task_state",
                )
                if key in snapshot.metadata
            },
        },
        "condition": condition,
        "language": language,
        "seed": int(seed),
        "policy_hash": policy_hash,
        "program_hash": program_hash,
        "model_id": model_id,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _plan_requests_position(plan: QueryPlan) -> bool:
    variables = tuple(
        str(value).casefold() for value in plan.target_variables
    )
    if any(
        token in variable
        for variable in variables
        for token in ("position", "location", "coordinate")
    ):
        return True
    text = plan.raw_text.casefold()
    return any(
        token in text
        for token in (
            "where",
            "position",
            "location",
            "coordinate",
            "哪里",
            "哪儿",
            "位置",
            "坐标",
        )
    )


def _cache_value(value: Any) -> Any:
    if is_dataclass(value):
        return _cache_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _cache_value(nested)
            for key, nested in value.items()
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_cache_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _model_cache_identity(backend: Any) -> str:
    configured = getattr(backend, "model_name_or_path", None)
    return str(
        configured
        or f"{type(backend).__module__}.{type(backend).__qualname__}"
    )


def _claims_and_verdicts_from_document(
    explanation_ir: ExplanationIR,
    document: ExplanationDocumentV3,
    *,
    frame: int | None,
) -> tuple[tuple[AtomicClaim, ...], tuple[ClaimVerdict, ...]]:
    """Bind displayed semantic units directly to their producing evidence."""

    by_id = {item.unit_id: item for item in explanation_ir.units}
    claims: list[AtomicClaim] = []
    verdicts: list[ClaimVerdict] = []
    for unit_id in document.used_unit_ids:
        unit = by_id.get(unit_id)
        if unit is None:
            continue
        claim = AtomicClaim(
            claim_id=unit.unit_id,
            # The online claim is the typed unit itself.  Section text may
            # combine multiple units and is retained in the document, not
            # duplicated into several allegedly atomic claims.
            text=unit.reference_text,
            claim_type=_claim_type_for_unit(unit.layer.value),
            entities=tuple(dict.fromkeys(unit.arguments)),
            frame_scope=(int(frame),) if frame is not None else (),
            time_scope="current decision",
            predicate=unit.predicate,
            expected_outcome=unit.value,
            modality="executed" if unit.trace_derived else "observed",
            confidence=1.0,
        )
        records = tuple(
            EvidenceRecord(
                evidence_id=evidence_id,
                source_type=_source_type_for_evidence(
                    evidence_id,
                    provenance=unit.provenance,
                ),
                frame_id=frame,
                program_branch_id=(
                    evidence_id if unit.trace_derived else None
                ),
                observed_value=unit.value,
                provenance={
                    "unit_id": unit.unit_id,
                    "predicate": unit.predicate,
                    "schema_version": "explanation-ir.v2",
                },
            )
            for evidence_id in unit.evidence_ids
        )
        claims.append(claim)
        verdicts.append(
            ClaimVerdict(
                claim=claim,
                status=ClaimVerdictStatus.SUPPORTED,
                evidence=records,
                confidence=1.0,
                verifier_reason=(
                    "The displayed span cites a typed ExplanationIR unit that "
                    "was compiled directly from these execution records."
                ),
            )
        )
    return tuple(claims), tuple(verdicts)


def _claim_type_for_unit(layer: str) -> str:
    return {
        "task_goal": "state",
        "policy_proposal": "action",
        "proposal_rationale": "causal",
        "coordination": "action",
        "final_action": "action",
        "counterfactual": "counterfactual",
    }.get(layer, "state")


def _source_type_for_evidence(
    evidence_id: str,
    *,
    provenance: str,
) -> str:
    if provenance == "program_trace" or evidence_id.startswith("program_"):
        return "program_trace"
    if provenance == "paired_simulation" or evidence_id.startswith("simulation"):
        return "simulation"
    if provenance == "neural_policy" or evidence_id == "neural_policy":
        return "neural_policy"
    return "state"


def _decision_causal_record(
    *,
    target: str,
    argmax_action: str,
    proposed_action: str | None,
    resolved_action: str | None,
    executed_action: str | None,
    action_resolution: Mapping[str, Any],
    observable_decision_context: Sequence[Mapping[str, Any]],
    execution_reason_context: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Bind the Actor proposal, arbitration, and final action together.

    Earlier explanation code received these as unrelated top-level fields.
    That allowed a generator to describe an Actor proposal or a generic goal
    condition while silently dropping the multi-agent relation that changed
    the executed action.  This typed record is the single decision-level
    source consumed by both explanation conditions.  It contains no authored
    answer text and contains no program audit fields; eligible trace evidence
    is attached separately only in the randomized treatment condition.
    """

    proposed = str(proposed_action or argmax_action)
    resolved = str(resolved_action or proposed)
    executed = str(executed_action or resolved)
    return {
        "schema": "decision_causal_record.v1",
        "target": str(target),
        "policy_decision": {
            "argmax_action": str(argmax_action),
            "proposed_action": proposed,
        },
        "joint_action_decision": {
            "resolved_action": resolved,
            "executed_action": executed,
            "changed_actor_proposal": proposed != executed,
            "resolution": dict(action_resolution),
        },
        "actor_observable_conditions": tuple(
            dict(item)
            for item in observable_decision_context
            if isinstance(item, Mapping)
        ),
        "recorded_execution_reasons": tuple(
            dict(item)
            for item in execution_reason_context
            if isinstance(item, Mapping)
        ),
    }


def _shared_task_context(
    state_facts: Sequence[Mapping[str, Any]],
    *,
    target: str,
) -> Mapping[str, Any]:
    """Return the same structured task evidence to both study conditions.

    The record is produced by the environment adapter's deterministic task
    logic.  It is neither an extracted-program trace nor a natural-language
    explanation.  Full authored sentences are removed recursively so each
    condition must realize the values itself.
    """

    for fact in state_facts:
        if str(fact.get("predicate", "")) != "objective_selection_reason":
            continue
        if target not in {
            str(value) for value in fact.get("arguments", ())
        }:
            continue
        value = fact.get("value", {})
        if not isinstance(value, Mapping):
            return {}
        return {
            "evidence_id": f"state::{fact.get('fact_id', target + '.objective_reason')}",
            "predicate": "objective_selection_reason",
            "target": target,
            "value": _remove_preauthored_text(value),
        }
    return {}


def _counterfactual_decision_contrast(
    adapter: EnvironmentAdapter,
    policy: PolicyProtocol,
    *,
    original_snapshot: EnvironmentSnapshot,
    edited_snapshot: EnvironmentSnapshot,
    target: str,
    original_distribution: Mapping[str, float],
    edited_distribution: Mapping[str, float],
    scene_edit: SceneEditResult | None,
    counterfactual: CounterfactualResult | None,
    program: ExecutableProgram | None = None,
    program_alignment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one typed before/after decision comparison.

    The language model receives values and relations, not a prewritten reason.
    Every item comes from the two restored decision states, their paired neural
    rollout, or (for Our Method) the executable extracted program.
    """

    if scene_edit is None or counterfactual is None:
        return {}

    first_pair = (
        counterfactual.paired_rollouts[0]
        if counterfactual.paired_rollouts
        else None
    )
    original_frame = (
        first_pair.baseline.frames[0]
        if first_pair is not None and first_pair.baseline.frames
        else None
    )
    edited_frame = (
        first_pair.counterfactual.frames[0]
        if first_pair is not None and first_pair.counterfactual.frames
        else None
    )

    def decision_side(
        frame: RolloutFrame | None,
        fallback_snapshot: EnvironmentSnapshot,
        fallback_distribution: Mapping[str, float],
    ) -> tuple[EnvironmentSnapshot, dict[str, Any]]:
        distribution = dict(fallback_distribution)
        snapshot = fallback_snapshot
        if frame is not None:
            actor_output = frame.distributions.get(target)
            if actor_output is not None:
                distribution = dict(
                    zip(actor_output.actions, actor_output.probabilities)
                )
            snapshot = replace(
                frame.snapshot,
                action_distributions=dict(frame.distributions),
                proposed_actions=dict(frame.proposed_actions),
                executed_actions=dict(frame.executed_actions),
                rewards=dict(frame.reward),
                metadata={
                    **dict(frame.snapshot.metadata),
                    "decision_evidence_aligned": True,
                    "action_resolution": dict(
                        frame.info.get("action_resolution", {})
                    ),
                    "environment_events": tuple(frame.environment_events),
                },
            )
        argmax_action = (
            max(distribution, key=distribution.__getitem__)
            if distribution
            else ""
        )
        proposed_action = str(
            snapshot.proposed_actions.get(target, argmax_action)
        )
        resolution = snapshot.metadata.get("action_resolution", {})
        target_resolution = (
            dict(resolution.get(target, {}))
            if isinstance(resolution, Mapping)
            and isinstance(resolution.get(target), Mapping)
            else {}
        )
        resolved_action = str(
            target_resolution.get("resolved_action", proposed_action)
        )
        executed_action = str(
            snapshot.executed_actions.get(target, resolved_action)
        )
        execution_facts = tuple(
            {
                "fact_id": fact.fact_id,
                "predicate": fact.predicate,
                "arguments": tuple(fact.arguments),
                "value": _remove_preauthored_text(fact.value),
                "factor_groups": tuple(fact.factor_groups),
            }
            for fact in adapter.neural_baseline_execution_facts(
                snapshot,
                target,
                policy,
            )
        )
        objective_facts = tuple(
            {
                "evidence_id": f"state::{fact.fact_id}",
                "fact_id": fact.fact_id,
                "predicate": fact.predicate,
                "arguments": tuple(fact.arguments),
                "value": _remove_preauthored_text(fact.value),
                "factor_groups": tuple(fact.factor_groups),
            }
            for fact in adapter.decision_objective_facts(
                snapshot,
                target,
                policy,
            )
        )
        return snapshot, {
            "action_distribution": distribution,
            "argmax_action": argmax_action,
            "proposed_action": proposed_action,
            "resolved_action": resolved_action,
            "executed_action": executed_action,
            "environment_changed_action": (
                proposed_action != executed_action
            ),
            "action_resolution": target_resolution,
            "execution_facts": execution_facts,
            "objective_context": (
                objective_facts[0] if objective_facts else {}
            ),
            "objective_contexts": objective_facts,
            "environment_events": tuple(
                event
                for event in snapshot.metadata.get(
                    "environment_events",
                    (),
                )
                if not isinstance(event, Mapping)
                or target in {
                    str(value)
                    for value in event.values()
                    if isinstance(value, (str, int, float))
                }
            )[:8],
        }

    original_decision_snapshot, original_side = decision_side(
        original_frame,
        original_snapshot,
        original_distribution,
    )
    edited_decision_snapshot, edited_side = decision_side(
        edited_frame,
        edited_snapshot,
        edited_distribution,
    )
    original_context = adapter.semantic_policy_context(
        original_decision_snapshot,
        target,
    )
    edited_context = adapter.semantic_policy_context(
        edited_decision_snapshot,
        target,
    )
    descriptions = adapter.semantic_feature_descriptions()
    intervention_entities = {
        str(item.entity_id) for item in scene_edit.interventions
    }
    reason_feature_map = adapter.action_constraint_reason_features()
    active_reason_features: set[str] = set()
    for context in (original_context, edited_context):
        for action, active_reasons in context.action_constraint_reasons.items():
            configured = reason_feature_map.get(action, {})
            active_reason_features.update(
                str(configured[reason])
                for reason in active_reasons
                if reason in configured
            )

    changed_features: list[tuple[int, str, dict[str, Any]]] = []
    relevant_actions = {
        str(original_side["proposed_action"]),
        str(original_side["executed_action"]),
        str(edited_side["proposed_action"]),
        str(edited_side["executed_action"]),
    }
    for feature in sorted(
        set(original_context.features) | set(edited_context.features)
    ):
        if feature.startswith("local."):
            continue
        original_value = float(original_context.features.get(feature, 0.0))
        edited_value = float(edited_context.features.get(feature, 0.0))
        original_bindings = dict(
            adapter.semantic_feature_entity_bindings(
                feature,
                original_context,
            )
        )
        edited_bindings = dict(
            adapter.semantic_feature_entity_bindings(
                feature,
                edited_context,
            )
        )
        if (
            abs(original_value - edited_value) <= 1e-9
            and original_bindings == edited_bindings
        ):
            continue
        bound_entities = set(original_bindings.values()) | set(
            edited_bindings.values()
        )
        components = set(feature.split("."))
        if feature in active_reason_features:
            priority = 0
        elif bound_entities.intersection(intervention_entities):
            priority = 0
        elif feature.startswith("candidate.") and components.intersection(
            relevant_actions
        ):
            priority = 1
        elif feature.startswith(("charger.", "other.")):
            priority = 2
        elif feature.startswith(("self.", "goal.")):
            priority = 3
        else:
            priority = 4
        changed_features.append(
            (
                priority,
                feature,
                {
                    "feature": feature,
                    "description": dict(descriptions.get(feature, {})),
                    "original_value": original_value,
                    "edited_value": edited_value,
                    "original_observed_meaning": dict(
                        adapter.semantic_feature_observation(
                            feature,
                            original_value,
                        )
                    ),
                    "edited_observed_meaning": dict(
                        adapter.semantic_feature_observation(
                            feature,
                            edited_value,
                        )
                    ),
                    "original_bound_entities": original_bindings,
                    "edited_bound_entities": edited_bindings,
                    "original_provenance": str(
                        original_context.feature_provenance.get(
                            feature,
                            "observation",
                        )
                    ),
                    "edited_provenance": str(
                        edited_context.feature_provenance.get(
                            feature,
                            "observation",
                        )
                    ),
                },
            )
        )
    changed_features.sort(key=lambda item: (item[0], item[1]))

    constraint_changes: list[dict[str, Any]] = []
    for action in adapter.action_schema():
        original_reasons = set(
            original_context.action_constraint_reasons.get(action, ())
        )
        edited_reasons = set(
            edited_context.action_constraint_reasons.get(action, ())
        )
        if original_reasons == edited_reasons:
            continue
        configured = reason_feature_map.get(action, {})
        reason_records = []
        for reason in sorted(original_reasons | edited_reasons):
            feature = str(configured.get(reason, ""))
            reason_records.append(
                {
                    "reason": reason,
                    "feature": feature,
                    "description": dict(descriptions.get(feature, {})),
                    "active_before": reason in original_reasons,
                    "active_after": reason in edited_reasons,
                    "original_observed_meaning": (
                        dict(
                            adapter.semantic_feature_observation(
                                feature,
                                float(
                                    original_context.features.get(
                                        feature,
                                        0.0,
                                    )
                                ),
                            )
                        )
                        if feature
                        else {}
                    ),
                    "edited_observed_meaning": (
                        dict(
                            adapter.semantic_feature_observation(
                                feature,
                                float(
                                    edited_context.features.get(
                                        feature,
                                        0.0,
                                    )
                                ),
                            )
                        )
                        if feature
                        else {}
                    ),
                    "original_bound_entities": dict(
                        adapter.semantic_feature_entity_bindings(
                            feature,
                            original_context,
                        )
                    ) if feature else {},
                    "edited_bound_entities": dict(
                        adapter.semantic_feature_entity_bindings(
                            feature,
                            edited_context,
                        )
                    ) if feature else {},
                }
            )
        constraint_changes.append(
            {
                "action": str(action),
                "removed_constraints": tuple(
                    sorted(original_reasons - edited_reasons)
                ),
                "added_constraints": tuple(
                    sorted(edited_reasons - original_reasons)
                ),
                "reason_records": tuple(reason_records),
            }
        )

    program_comparison: dict[str, Any] = {}
    if program is not None:
        original_program = program.execute(
            original_context.features,
            original_context,
        )
        edited_program = program.execute(
            edited_context.features,
            edited_context,
        )

        def compact_program_side(execution: Any) -> dict[str, Any]:
            return {
                "action": execution.action,
                "path": tuple(
                    {
                        "feature": step.feature,
                        "observed_value": float(step.observed_value),
                        "operator": step.operator,
                        "threshold": float(step.threshold),
                        "result": bool(step.result),
                        "description": dict(
                            descriptions.get(step.feature, {})
                        ),
                        "bound_entities": dict(
                            adapter.semantic_feature_entity_bindings(
                                step.feature,
                                (
                                    original_context
                                    if execution is original_program
                                    else edited_context
                                ),
                            )
                        ),
                    }
                    for step in execution.trace.tree_steps
                ),
                "excluded_actions": tuple(
                    {
                        "action": item.action,
                        "active_reason_features": tuple(
                            item.active_reason_features
                        ),
                        "bound_entities": dict(item.bound_entities),
                    }
                    for item in execution.trace.excluded_actions
                ),
            }

        program_comparison = {
            "eligible_for_causal_explanation": bool(
                program_alignment
                and program_alignment.get("direction_consistent", False)
            ),
            "original": compact_program_side(original_program),
            "edited": compact_program_side(edited_program),
            "alignment": dict(program_alignment or {}),
        }

    def objective_signature(side: Mapping[str, Any]) -> Any:
        context = side.get("objective_context", {})
        if not isinstance(context, Mapping):
            return None
        value = context.get("value", {})
        if not isinstance(value, Mapping):
            return None
        selected = value.get("selected_objective", {})
        return (
            _remove_preauthored_text(selected)
            if isinstance(selected, Mapping)
            else selected
        )

    original_objective = objective_signature(original_side)
    edited_objective = objective_signature(edited_side)
    objective_changed = (
        original_objective is not None
        and edited_objective is not None
        and original_objective != edited_objective
    )

    return {
        "target": target,
        "required_explanation_components": (
            "original.executed_action",
            "original.objective_context",
            "original_decision_condition",
            "intervention",
            "changed_observable_factor_or_constraint",
            "edited.objective_context",
            "edited.executed_action",
        ),
        "interventions": tuple(
            asdict(item) for item in scene_edit.interventions
        ),
        "original": original_side,
        "edited": edited_side,
        "action_changed": (
            original_side["executed_action"]
            != edited_side["executed_action"]
        ),
        "objective_changed": objective_changed,
        "changed_observable_factors": tuple(
            item[2] for item in changed_features[:24]
        ),
        "constraint_changes": tuple(constraint_changes),
        "program_comparison": program_comparison,
        "paired_rollout_count": len(counterfactual.paired_rollouts),
    }


def _remove_preauthored_text(value: Any) -> Any:
    """Keep typed evidence while removing complete authored prose fields."""

    if isinstance(value, Mapping):
        return {
            str(key): _remove_preauthored_text(item)
            for key, item in value.items()
            if str(key) not in {"verbalizations", "fact_verbalizations"}
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_remove_preauthored_text(item) for item in value)
    return value


def _first_distributions(rollout: Any) -> dict[str, dict[str, float]]:
    if not rollout.frames:
        return {}
    return {
        agent_id: dict(zip(distribution.actions, distribution.probabilities))
        for agent_id, distribution in rollout.frames[0].distributions.items()
    }


def _brief_exception(error: BaseException, *, limit: int = 280) -> str:
    return " ".join(str(error).split())[:limit]


def _program_trace_is_groundable(
    trace: Sequence[Mapping[str, Any]],
) -> bool:
    """Match the shared generator's typed trace-admission contract."""

    for item in trace:
        trace_type = str(item.get("trace_type", ""))
        if trace_type == "tree_branch":
            meaning = item.get("observed_meaning", {})
            if isinstance(meaning, Mapping) and meaning.get(
                "explanation_role"
            ):
                return True
        elif trace_type == "action_constraint" and item.get(
            "active_reason_features"
        ):
            return True
    return False


def _frame_evidence(frame: Any) -> dict[str, Any]:
    actor_outputs = {
        agent_id: distribution_evidence(distribution)
        for agent_id, distribution in frame.distributions.items()
    }
    return {
        "frame": frame.frame,
        "actor_outputs": actor_outputs,
        "proposed_actions": dict(frame.proposed_actions),
        "executed_actions": dict(frame.executed_actions),
        "action_masks": {
            agent_id: tuple(mask)
            for agent_id, mask in frame.action_masks.items()
        },
        "reward": dict(frame.reward),
        "reward_breakdown": dict(frame.reward_breakdown),
        "task_state": dict(frame.task_state),
        "charging_state": dict(frame.charging_state),
        "environment_events": list(frame.environment_events),
        "environment_enforced_action_changes": {
            agent_id: (
                frame.proposed_actions.get(agent_id)
                != frame.executed_actions.get(agent_id)
            )
            for agent_id in frame.proposed_actions
        },
        "next_frame": (
            frame.next_snapshot.frame
            if frame.next_snapshot is not None
            else None
        ),
        "next_state": (
            _plain_evidence_value(frame.next_snapshot.state)
            if frame.next_snapshot is not None
            else None
        ),
        "done": frame.done,
        "info": dict(frame.info),
    }


def distribution_evidence(distribution: Any) -> dict[str, Any]:
    """Serialize one actor output into immutable, JSON-safe evidence.

    This helper is part of the boundary between the query engine and the
    extracted evidence module.  Keeping it public prevents callers from
    depending on an unimported module-private implementation after evidence
    orchestration is split across files.
    """

    return {
        "actions": tuple(distribution.actions),
        "raw_logits": tuple(float(value) for value in distribution.logits),
        "action_mask": tuple(float(value) for value in distribution.action_mask),
        "masked_probabilities": {
            action: float(probability)
            for action, probability in zip(
                distribution.actions,
                distribution.probabilities,
            )
        },
        "proposed_action": distribution.proposed_action,
        "argmax_action": distribution.argmax_action,
    }


def _plain_evidence_value(value: Any) -> Any:
    return asdict(value) if is_dataclass(value) else value


def _rollout_total_reward(rollout: Any) -> dict[str, float]:
    totals: dict[str, float] = {}
    for frame in rollout.frames:
        for agent_id, value in frame.reward.items():
            totals[agent_id] = totals.get(agent_id, 0.0) + float(value)
    return totals


def _desired_action(
    plan: QueryPlan,
    target: str,
    action_names: tuple[str, ...],
) -> str | None:
    candidates = tuple(
        value
        for key, value in plan.desired_outcomes.items()
        if (
            str(key).split(".", 1)[0] in {target, str(key)}
            and "action" in str(key).rsplit(".", 1)[-1].casefold()
        )
    ) + (plan.desired_outcomes.get("action"),)
    for value in candidates:
        if value is None:
            continue
        action = str(value).upper()
        if action in action_names:
            return action
    return None


def _desired_objective(
    plan: QueryPlan,
    target: str,
    objective_names: tuple[str, ...],
) -> str | None:
    allowed = {str(value) for value in objective_names}
    for key, value in plan.desired_outcomes.items():
        path = str(key)
        leaf = path.rsplit(".", 1)[-1].casefold()
        owner = path.split(".", 1)[0] if "." in path else target
        if owner != target or leaf not in {
            "objective",
            "goal",
            "goal_kind",
            "task",
        }:
            continue
        objective = str(value)
        if not allowed or objective in allowed:
            return objective
    return None


def _candidate_effect_evidence(
    effect: Any,
    program_alignment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = effect.counterfactual
    return {
        "candidate_id": effect.candidate.candidate_id,
        "factor": effect.candidate.factor,
        "description": effect.candidate.description,
        "interventions": [
            asdict(item) for item in effect.candidate.interventions
        ],
        "provenance": dict(effect.candidate.provenance),
        "observable_by_actor": effect.observable_by_actor,
        "l1_policy_effect": effect.l1_policy_effect,
        "action_probability_delta": dict(effect.action_probability_delta),
        "action_change_rate": effect.action_change_rate,
        "supported": effect.supported,
        "paired_seeds": tuple(result.paired_seeds),
        "baseline_first_action_distribution": dict(
            result.baseline_first_action_distribution
        ),
        "counterfactual_first_action_distribution": dict(
            result.first_action_distribution
        ),
        "program_counterfactual_alignment": dict(
            program_alignment or {}
        ),
        "causal_claim_allowed": bool(
            effect.supported
            and effect.observable_by_actor
            and (
                program_alignment is None
                or program_alignment.get("direction_consistent", False)
            )
        ),
    }


def _local_program_reliability(
    program: ExecutableProgram,
    *,
    scenario_tags: tuple[str, ...],
    neural_action: str,
    program_action: str,
) -> dict[str, Any]:
    """Audit whether a tree path may explain this particular NN decision.

    Global action fidelity can hide a weak interaction branch. The program is
    explanation-eligible only when it agrees on the current action and every
    current non-ordinary scenario group reaches the held-out fidelity required
    by the extraction configuration. This is a local explanation gate, not a
    requirement that the tree replace the NN in a rollout.
    """

    metadata = (
        program.metadata
        if isinstance(program.metadata, Mapping)
        else {}
    )
    metrics = metadata.get("metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}
    config = metadata.get("config", {})
    if not isinstance(config, Mapping):
        config = {}
    group_fidelity = metrics.get("group_action_fidelity", {})
    if not isinstance(group_fidelity, Mapping):
        group_fidelity = {}
    group_samples = metrics.get("group_validation_samples", {})
    if not isinstance(group_samples, Mapping):
        group_samples = {}
    overall_threshold = float(
        config.get(
            "minimum_overall_fidelity_for_explanation",
            0.85,
        )
    )
    if overall_threshold <= 0.0:
        overall_threshold = 0.85
    maximum_explanation_kl_raw = config.get(
        "maximum_mean_kl_for_explanation",
        0.25,
    )
    maximum_explanation_kl = (
        0.25
        if maximum_explanation_kl_raw is None
        else max(0.0, float(maximum_explanation_kl_raw))
    )
    threshold = float(
        config.get(
            "minimum_interaction_fidelity_for_explanation",
            config.get(
                "minimum_interaction_fidelity_for_feedback",
                0.75,
            ),
        )
    )
    if threshold <= 0.0:
        # Core compatibility defaults are permissive, but a user-facing
        # explanation needs a real held-out quality floor.
        threshold = 0.75
    minimum_samples = max(
        0,
        int(config.get("minimum_interaction_validation_samples", 0)),
    )
    counterfactual_threshold = max(
        0.0,
        float(
            config.get(
                "minimum_counterfactual_direction_fidelity_for_explanation",
                config.get(
                    "minimum_counterfactual_direction_fidelity_for_feedback",
                    0.0,
                ),
            )
        ),
    )
    minimum_changed_pairs = max(
        0,
        int(
            config.get(
                "minimum_counterfactual_changed_pairs_for_explanation",
                config.get("minimum_counterfactual_changed_pairs", 0),
            )
        ),
    )
    minimum_counterfactual_pairs = max(
        0,
        int(config.get("minimum_counterfactual_pairs", 0)),
    )
    relevant_groups = tuple(
        dict.fromkeys(
            str(tag)
            for tag in scenario_tags
            if str(tag) and str(tag) != "ordinary"
        )
    )
    reasons: list[str] = []
    if neural_action != program_action:
        reasons.append("current_action_disagreement")
    overall_fidelity = metrics.get("action_fidelity")
    if overall_fidelity is None:
        reasons.append("unvalidated_overall_fidelity")
    elif float(overall_fidelity) <= overall_threshold:
        reasons.append("overall_fidelity_below_explanation_threshold")
    mean_kl = metrics.get("mean_kl_divergence")
    if mean_kl is None:
        reasons.append("unvalidated_program_fit_kl")
    elif float(mean_kl) >= maximum_explanation_kl:
        reasons.append("program_fit_kl_above_explanation_threshold")
    evaluated_groups: dict[str, dict[str, Any]] = {}
    for group in relevant_groups:
        score = group_fidelity.get(group)
        samples = group_samples.get(group)
        evaluated_groups[group] = {
            "fidelity": float(score) if score is not None else None,
            "validation_samples": (
                int(samples) if samples is not None else None
            ),
        }
        if score is None:
            reasons.append(f"unvalidated_scenario_group:{group}")
            continue
        if samples is not None and int(samples) < minimum_samples:
            reasons.append(f"insufficient_scenario_samples:{group}")
        if float(score) < threshold:
            reasons.append(
                f"scenario_fidelity_below_threshold:{group}"
            )
    changed_pairs = int(
        metrics.get("counterfactual_changed_pairs", 0) or 0
    )
    dataset_pairs = int(
        metrics.get("counterfactual_dataset_pairs", 0) or 0
    )
    direction_fidelity = metrics.get(
        "counterfactual_direction_fidelity"
    )
    if dataset_pairs < minimum_counterfactual_pairs:
        reasons.append("insufficient_counterfactual_pairs")
    if minimum_changed_pairs > 0:
        if changed_pairs < minimum_changed_pairs:
            reasons.append("insufficient_counterfactual_changed_pairs")
        elif (
            direction_fidelity is None
            or float(direction_fidelity) <= counterfactual_threshold
        ):
            reasons.append(
                "counterfactual_direction_fidelity_below_threshold"
            )
    return {
        "reliable": not reasons,
        "reasons": tuple(reasons),
        "scenario_tags": tuple(str(tag) for tag in scenario_tags),
        "required_group_fidelity": threshold,
        "overall_fidelity": (
            float(overall_fidelity)
            if overall_fidelity is not None
            else None
        ),
        "required_overall_fidelity": overall_threshold,
        "program_fit_kl": float(mean_kl) if mean_kl is not None else None,
        "maximum_explanation_kl": maximum_explanation_kl,
        "required_group_samples": minimum_samples,
        "evaluated_groups": evaluated_groups,
        "counterfactual_direction_fidelity": (
            float(direction_fidelity)
            if direction_fidelity is not None
            else None
        ),
        "counterfactual_changed_pairs": changed_pairs,
        "counterfactual_dataset_pairs": dataset_pairs,
        "required_counterfactual_direction_fidelity": (
            counterfactual_threshold
        ),
        "required_counterfactual_changed_pairs": minimum_changed_pairs,
        "required_counterfactual_pairs": minimum_counterfactual_pairs,
    }


def _program_counterfactual_alignment(
    program: ExecutableProgram,
    adapter: EnvironmentAdapter,
    result: CounterfactualResult,
    target_entity: str,
) -> dict[str, Any]:
    """Compare NN and program probability-change vectors on one legal pair."""

    baseline_context = adapter.semantic_policy_context(
        result.original_snapshot,
        target_entity,
    )
    edited_context = adapter.semantic_policy_context(
        result.intervened_snapshot,
        target_entity,
    )
    try:
        program_baseline = program.execute(
            baseline_context.features,
            baseline_context,
        )
        program_edited = program.execute(
            edited_context.features,
            edited_context,
        )
    except Exception as exc:
        return {
            "available": False,
            "direction_consistent": False,
            "audit_error": (
                f"{type(exc).__name__}: {_brief_exception(exc)}"
            ),
        }
    actions = tuple(program.action_names)
    neural_baseline_map = result.baseline_first_action_distribution.get(
        target_entity,
        {},
    )
    neural_edited_map = result.first_action_distribution.get(
        target_entity,
        {},
    )
    neural_baseline = np.asarray(
        [float(neural_baseline_map.get(action, 0.0)) for action in actions],
        dtype=float,
    )
    neural_edited = np.asarray(
        [float(neural_edited_map.get(action, 0.0)) for action in actions],
        dtype=float,
    )
    program_baseline_values = np.asarray(
        [float(program_baseline.probabilities[action]) for action in actions],
        dtype=float,
    )
    program_edited_values = np.asarray(
        [float(program_edited.probabilities[action]) for action in actions],
        dtype=float,
    )
    neural_delta = neural_edited - neural_baseline
    program_delta = program_edited_values - program_baseline_values
    denominator = float(
        np.linalg.norm(neural_delta) * np.linalg.norm(program_delta)
    )
    cosine = (
        float(np.dot(neural_delta, program_delta) / denominator)
        if denominator > 1e-12
        else None
    )
    neural_before = actions[int(np.argmax(neural_baseline))]
    neural_after = actions[int(np.argmax(neural_edited))]
    program_before = program_baseline.action
    program_after = program_edited.action
    neural_changed = neural_before != neural_after
    program_changed = program_before != program_after
    direction_consistent = bool(
        cosine is not None
        and cosine >= 0.5
        and neural_changed == program_changed
        and (not neural_changed or neural_after == program_after)
    )
    return {
        "available": True,
        "neural_baseline_action": neural_before,
        "neural_counterfactual_action": neural_after,
        "program_baseline_action": program_before,
        "program_counterfactual_action": program_after,
        "baseline_action_agreement": neural_before == program_before,
        "counterfactual_action_agreement": neural_after == program_after,
        "neural_probability_delta": dict(
            zip(actions, neural_delta.tolist())
        ),
        "program_probability_delta": dict(
            zip(actions, program_delta.tolist())
        ),
        "delta_cosine_similarity": cosine,
        "direction_consistent": direction_consistent,
    }


def _why_analysis_evidence(
    analysis: Any,
    program_alignments: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    alignments = dict(program_alignments or {})
    supported = tuple(
        item
        for item in analysis.supported_candidates
        if not alignments
        or alignments.get(
            item.candidate.candidate_id,
            {},
        ).get("direction_consistent", False)
    )
    return {
        "target_entity": analysis.target_entity,
        "baseline_action": analysis.baseline_action,
        "minimum_effect": analysis.minimum_effect,
        "supported": bool(supported),
        "supported_candidate_ids": tuple(
            item.candidate.candidate_id
            for item in supported
        ),
        "candidates": [
            _candidate_effect_evidence(
                item,
                alignments.get(item.candidate.candidate_id),
            )
            for item in analysis.candidates
        ],
    }


def _why_not_evidence(recourse: Any) -> dict[str, Any]:
    return {
        "target_entity": recourse.target_entity,
        "desired_action": recourse.desired_action,
        "baseline_probability": recourse.baseline_probability,
        "achieved": recourse.achieved,
        "selected_candidate_id": (
            recourse.selected.candidate.candidate_id
            if recourse.selected is not None
            else None
        ),
        "selected": (
            _candidate_effect_evidence(recourse.selected)
            if recourse.selected is not None
            else None
        ),
        "refusal_reason": recourse.refusal_reason,
        "candidates": [
            _candidate_effect_evidence(item)
            for item in recourse.candidates
        ],
    }
