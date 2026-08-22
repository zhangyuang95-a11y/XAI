"""General open-ended Warehouse query execution and evidence orchestration."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from enum import Enum
import hashlib
import json
import time
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from backend.adapters.base import (
    EnvironmentAdapter,
    EnvironmentSnapshot,
    Intervention,
    PolicyProtocol,
    RolloutFrame,
)
from core.program import ExecutableProgram
from backend.nlp.explanation_generator import ExecutionGroundedExplanationGenerator
from backend.nlp.explanation_ir import (
    ExplanationDocumentV3,
    ExplanationIR,
    validate_document,
)
from backend.nlp.semantic_query_planner import (
    SemanticTransformerQueryPlanner as TransformerQueryPlanner,
)
from backend.nlp.schemas import (
    AtomicClaim,
    ClaimVerdict,
    EvidenceBundle,
    QueryIntent,
    QueryPlan,
)
from backend.simulation.counterfactual import CounterfactualEngine, CounterfactualResult
from backend.simulation.intervention import InterventionEngine, SceneEditResult


class ExplanationCondition(str, Enum):
    """The randomized treatment in the single-variable trace ablation."""

    NO_TRACE = "no_trace"
    RCPD_TRACE = "rcpd_trace"


EXPLANATION_MODE_NO_TRACE = ExplanationCondition.NO_TRACE.value
EXPLANATION_MODE_RCPD_TRACE = ExplanationCondition.RCPD_TRACE.value
EXPLANATION_MODES = frozenset(item.value for item in ExplanationCondition)


@dataclass(frozen=True)
class QueryAnswer:
    query_plan: QueryPlan
    evidence: EvidenceBundle
    explanation: str
    claims: tuple[AtomicClaim, ...]
    verdicts: tuple[ClaimVerdict, ...]
    explanation_mode: str = EXPLANATION_MODE_RCPD_TRACE
    generation_grounding: Mapping[str, Any] | None = None
    scene_edit: SceneEditResult | None = None
    counterfactual: CounterfactualResult | None = None
    posthoc_warnings: tuple[str, ...] = ()
    display_explanation: str | None = None
    display_claims: tuple[AtomicClaim, ...] | None = None
    display_verdicts: tuple[ClaimVerdict, ...] | None = None
    raw_explanation: str | None = None
    raw_claims: tuple[AtomicClaim, ...] | None = None
    raw_verdicts: tuple[ClaimVerdict, ...] | None = None
    trace_audit: Mapping[str, Any] | None = None
    explanation_document: Mapping[str, Any] | None = None
    explanation_ir_hash: str | None = None
    generation_diagnostics: Mapping[str, Any] | None = None

    @property
    def user_visible_explanation(self) -> str:
        """Return only text that passed the post-generation display gate."""

        return (
            self.explanation
            if self.display_explanation is None
            else self.display_explanation
        )

    @property
    def user_visible_verdicts(self) -> tuple[ClaimVerdict, ...]:
        """Hide rejected raw-generation Claims from the explanation panel."""

        if self.display_verdicts is not None:
            return self.display_verdicts
        if self.display_explanation is None:
            return self.verdicts
        return tuple(
            verdict
            for verdict in self.verdicts
            if verdict.status.value == "SUPPORTED"
        )

    @property
    def user_visible_claims(self) -> tuple[AtomicClaim, ...]:
        if self.display_claims is not None:
            return self.display_claims
        if self.display_explanation is None:
            return self.claims
        return tuple(verdict.claim for verdict in self.user_visible_verdicts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_plan": self.query_plan.to_dict(),
            "evidence": self.evidence.to_dict(),
            "explanation": self.user_visible_explanation,
            "raw_explanation": (
                self.explanation
                if self.raw_explanation is None
                else self.raw_explanation
            ),
            "explanation_mode": self.explanation_mode,
            "claims": [
                claim.to_dict() for claim in self.user_visible_claims
            ],
            "verdicts": [
                verdict.to_dict()
                for verdict in self.user_visible_verdicts
            ],
            "raw_claims": [
                claim.to_dict()
                for claim in (
                    self.claims
                    if self.raw_claims is None
                    else self.raw_claims
                )
            ],
            "raw_verdicts": [
                verdict.to_dict()
                for verdict in (
                    self.verdicts
                    if self.raw_verdicts is None
                    else self.raw_verdicts
                )
            ],
            "trace_audit": dict(self.trace_audit or {}),
            "explanation_document": dict(self.explanation_document or {}),
            "explanation_ir_hash": self.explanation_ir_hash,
            "generation_diagnostics": dict(self.generation_diagnostics or {}),
            "latency_ms": (
                float(self.generation_diagnostics.get("total_ms", 0.0))
                if self.generation_diagnostics
                else None
            ),
            "generation_grounding": dict(
                self.generation_grounding or {}
            ),
            "scene_edit": (
                {
                    "interventions": [
                        asdict(item) for item in self.scene_edit.interventions
                    ],
                    "changes": [asdict(item) for item in self.scene_edit.changes],
                }
                if self.scene_edit
                else None
            ),
            "posthoc_warnings": list(self.posthoc_warnings),
        }


@dataclass(frozen=True)
class ManualCounterfactualAnswer:
    validation: Any
    counterfactual: CounterfactualResult | None


@dataclass(frozen=True)
class _CachedRealization:
    document: ExplanationDocumentV3
    raw_document: ExplanationDocumentV3 | None
    raw_text: str | None


class WarehouseQueryEngine:
    """Execute questions with the NN controller and a read-only program audit.

    ``policy`` is the sole rollout/controller policy. ``program`` may be
    executed over audited semantic contexts to expose a local rule trace and
    alignment diagnostics; it is never passed to the environment or
    counterfactual engine.
    """

    def __init__(
        self,
        *,
        adapter: EnvironmentAdapter,
        policy: PolicyProtocol,
        planner: TransformerQueryPlanner,
        explanation_generator: ExecutionGroundedExplanationGenerator,
        program: ExecutableProgram | None = None,
        policy_artifact_hash: str | None = None,
        program_artifact_hash: str | None = None,
    ) -> None:
        self.adapter = adapter
        self.policy = policy
        self.planner = planner
        self.explanation_generator = explanation_generator
        if self.explanation_generator.semantics is None:
            self.explanation_generator.semantics = adapter
        self.program = program
        self.policy_artifact_hash = policy_artifact_hash
        self.program_artifact_hash = program_artifact_hash
        self.interventions = InterventionEngine(adapter)
        # Causal and what-if evidence must measure the deployed neural Actor.
        # The extracted program is audited separately and never substitutes
        # for the NN in a counterfactual rollout.
        self.counterfactuals = CounterfactualEngine(
            adapter,
            policy,
        )
        # Documents are immutable and keyed by their complete semantic input.
        # Keep this cache for the lifetime of the session engine; rebuilding it
        # inside execute_plan would make every request an artificial miss.
        self._document_cache: OrderedDict[str, _CachedRealization] = (
            OrderedDict()
        )
        self._document_cache_size = 512
        self._ir_cache: OrderedDict[str, ExplanationIR] = OrderedDict()
        self._ir_cache_size = 512
        self._answer_cache: OrderedDict[str, QueryAnswer] = OrderedDict()
        self._answer_cache_size = 256

    def advance(
        self,
        snapshot: EnvironmentSnapshot,
        *,
        deterministic: bool = False,
        seed: int | None = None,
    ) -> RolloutFrame | None:
        """Advance every Agent once from a selected frame."""

        self.adapter.restore(snapshot, self.policy)
        if seed is not None and not deterministic and hasattr(
            self.policy, "seed_rng"
        ):
            self.policy.seed_rng(seed)  # type: ignore[attr-defined]
        rollout = self.adapter.rollout(
            self.policy,
            horizon=1,
            deterministic=deterministic,
        )
        return rollout.frames[0] if rollout.frames else None

    def execute_manual_interventions(
        self,
        snapshot: EnvironmentSnapshot,
        interventions: tuple[Intervention, ...],
        *,
        horizon: int,
        repetitions: int,
        seed: int,
    ) -> ManualCounterfactualAnswer:
        """Backend API for the structured editor; commits nothing on failure."""

        validation = self.interventions.validate(
            snapshot,
            interventions,
        )
        if not validation.valid:
            return ManualCounterfactualAnswer(validation, None)
        result = self.counterfactuals.simulate(
            snapshot,
            interventions,
            horizon=max(1, horizon),
            repetitions=max(2, repetitions),
            deterministic=False,
            seed=seed,
        )
        return ManualCounterfactualAnswer(validation, result)

    def answer(
        self,
        question: str,
        snapshot: EnvironmentSnapshot,
        *,
        selected_frame: int | None = None,
        language: str = "auto",
        seed: int = 2026,
        snapshot_resolver: Callable[
            [int], EnvironmentSnapshot
        ] | None = None,
        on_explanation: Callable[[str], None] | None = None,
        explanation_mode: str = EXPLANATION_MODE_RCPD_TRACE,
    ) -> QueryAnswer:
        plan = self.planner.parse(
            question,
            selected_frame=(
                snapshot.frame if selected_frame is None else selected_frame
            ),
            environment_schema={
                "observations": dict(self.adapter.observation_schema()),
                "actions": list(self.adapter.action_schema()),
                "entities": dict(self.adapter.entity_schema()),
                **dict(
                    self.adapter.question_vocabulary()
                    if hasattr(self.adapter, "question_vocabulary")
                    else {}
                ),
                "focus_entity": self.adapter.default_target_entity(snapshot),
            },
            cache_context=self.question_cache_context(snapshot),
        )
        if plan.clarification_required:
            raise ValueError(
                plan.clarification_reason
                or "Please clarify the target entity and requested change."
            )
        return self.execute_plan(
            plan,
            snapshot,
            language=(
                plan.response_language
                if language.lower() in {"auto", "und"}
                else language
            ),
            seed=seed,
            snapshot_resolver=snapshot_resolver,
            on_explanation=on_explanation,
            explanation_mode=explanation_mode,
            _parse_diagnostics=dict(
                getattr(self.planner, "last_diagnostics", {})
            ),
        )

    def question_cache_context(
        self,
        snapshot: EnvironmentSnapshot,
    ) -> Mapping[str, Any]:
        """Return artifact and environment identities for QuestionIR caching."""

        state_payload = json.dumps(
            {
                "environment": snapshot.environment,
                "frame": snapshot.frame,
                "state": _cache_value(snapshot.state),
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
        return {
            "schema_version": "question-ir.v2",
            "checkpoint_sha256": self.policy_artifact_hash,
            "program_sha256": self.program_artifact_hash,
            "environment_id": (
                f"{type(self.adapter).__module__}."
                f"{type(self.adapter).__qualname__}"
            ),
            "evidence_state_sha256": hashlib.sha256(state_payload).hexdigest(),
        }

    def precompile_frame_irs(
        self,
        snapshot: EnvironmentSnapshot,
        *,
        languages: Sequence[str] = ("zh-CN", "en"),
    ) -> int:
        """Compile ordinary per-entity decision IRs without generating text."""

        compiled = 0
        for target in self.adapter.policy_entity_references():
            try:
                facts = self.adapter.evidence_facts(
                    snapshot,
                    target,
                    self.policy,
                )
                state_facts = tuple(
                    {
                        "fact_id": fact.fact_id,
                        "predicate": fact.predicate,
                        "arguments": fact.arguments,
                        "value": fact.value,
                        "factor_groups": fact.factor_groups,
                        "verbalizations": fact.verbalizations,
                        "value_verbalizations": fact.value_verbalizations,
                    }
                    for fact in facts
                )
                distribution = self.adapter.policy_distribution(
                    snapshot,
                    target,
                    self.policy,
                )
                proposed = str(
                    snapshot.proposed_actions.get(target)
                    or max(distribution, key=distribution.__getitem__)
                )
                final = str(
                    snapshot.executed_actions.get(target)
                    or proposed
                )
                resolutions = snapshot.metadata.get(
                    "action_resolution",
                    {},
                )
                resolution = (
                    dict(resolutions.get(target, {}))
                    if isinstance(resolutions, Mapping)
                    and isinstance(resolutions.get(target), Mapping)
                    else {}
                )
                shared_task_context = _shared_task_context(
                    state_facts,
                    target=target,
                )
                plan = QueryPlan(
                    raw_text="",
                    intent=QueryIntent.EXPLANATORY,
                    frame_reference=snapshot.frame,
                    subjects=(target,),
                    requires_policy_query=True,
                    requires_program_trace=True,
                    target_variables=(f"{target}.observed_action",),
                )
                evidence = EvidenceBundle(
                    query_plan=plan,
                    direct_result={
                        "target": target,
                        "proposed_action": proposed,
                        "executed_action": final,
                        "action_resolution": resolution,
                        "shared_task_context": shared_task_context,
                    },
                    state_facts=state_facts,
                    policy_results={
                        "target": target,
                        "proposed_action": proposed,
                        "executed_action": final,
                        "action_resolution": resolution,
                        "shared_task_context": shared_task_context,
                    },
                )
                for language in languages:
                    key = _common_ir_cache_key(
                        evidence,
                        language=language,
                        policy_hash=self.policy_artifact_hash,
                        program_hash=self.program_artifact_hash,
                        environment_id=(
                            f"{type(self.adapter).__module__}."
                            f"{type(self.adapter).__qualname__}"
                        ),
                    )
                    if key in self._ir_cache:
                        continue
                    self._ir_cache[key] = (
                        self.explanation_generator.compiler.compile_common(
                            evidence,
                            requested_language=language,
                            semantics=self.explanation_generator.semantics,
                        )
                    )
                    compiled += 1
            except (KeyError, RuntimeError, TypeError, ValueError):
                # Prefetch is an optimization and can never block a frame.
                continue
        while len(self._ir_cache) > self._ir_cache_size:
            self._ir_cache.popitem(last=False)
        return compiled

    def execute_plan(
        self,
        plan: QueryPlan,
        snapshot: EnvironmentSnapshot,
        *,
        language: str = "auto",
        seed: int = 2026,
        snapshot_resolver: Callable[
            [int], EnvironmentSnapshot
        ] | None = None,
        on_explanation: Callable[[str], None] | None = None,
        explanation_mode: str = EXPLANATION_MODE_RCPD_TRACE,
        _parse_diagnostics: Mapping[str, Any] | None = None,
    ) -> QueryAnswer:
        """Execute the exact plan shown to the user in a plan-preview UI."""

        execution_started = time.perf_counter()

        plan_errors = plan.validate()
        if plan_errors:
            raise ValueError(
                "Refusing to execute an invalid QueryPlan: "
                + "; ".join(plan_errors)
            )

        try:
            condition = (
                explanation_mode
                if isinstance(explanation_mode, ExplanationCondition)
                else ExplanationCondition(str(explanation_mode))
            )
        except ValueError as exc:
            raise ValueError(
                "Unknown explanation mode "
                f"{explanation_mode!r}; expected one of "
                f"{sorted(EXPLANATION_MODES)}."
            ) from exc
        include_program_trace = condition is ExplanationCondition.RCPD_TRACE

        frame_resolution_error: str | None = None
        if (
            plan.frame_reference is not None
            and plan.frame_reference != snapshot.frame
        ):
            if snapshot_resolver is None:
                frame_resolution_error = (
                    "QueryPlan frame does not match the supplied checkpoint "
                    f"({plan.frame_reference} != {snapshot.frame})."
                )
            else:
                try:
                    snapshot = snapshot_resolver(
                        plan.frame_reference
                    )
                except (IndexError, KeyError, ValueError) as exc:
                    frame_resolution_error = (
                        f"Requested frame {plan.frame_reference} could not "
                        f"be restored: {exc}"
                    )

        answer_cache_key = _answer_cache_key(
            plan,
            snapshot,
            condition=condition.value,
            language=language,
            seed=seed,
            policy_hash=self.policy_artifact_hash,
            program_hash=self.program_artifact_hash,
            model_id=_model_cache_identity(
                self.planner.backend
            ),
        )
        if answer_cache_key in self._answer_cache:
            cached = self._answer_cache.pop(answer_cache_key)
            self._answer_cache[answer_cache_key] = cached
            parse_diagnostics = dict(_parse_diagnostics or {})
            diagnostics = {
                **dict(cached.generation_diagnostics or {}),
                "parse_ms": float(parse_diagnostics.get("latency_ms", 0.0)),
                "simulation_ms": 0.0,
                "ir_compile_ms": 0.0,
                "realization_ms": 0.0,
                "validation_ms": 0.0,
                "total_ms": (time.perf_counter() - execution_started) * 1000.0,
                "model_call_count": int(
                    parse_diagnostics.get("model_call_count", 0)
                ),
                "question_ir_model_calls": int(
                    parse_diagnostics.get("model_call_count", 0)
                ),
                "realization_model_calls": 0,
                "question_cache_hit": bool(
                    parse_diagnostics.get("cache_hit", False)
                ),
                "ir_cache_hit": True,
                "document_cache_hit": True,
                "answer_cache_hit": True,
                "parse_input_tokens": int(
                    parse_diagnostics.get("input_tokens", 0)
                ),
                "parse_output_tokens": int(
                    parse_diagnostics.get("output_tokens", 0)
                ),
                "realization_input_tokens": 0,
                "realization_output_tokens": 0,
                "question_binding": dict(
                    parse_diagnostics.get("binding", {})
                ),
            }
            answer = replace(
                cached,
                generation_diagnostics=diagnostics,
            )
            if on_explanation is not None:
                on_explanation(answer.user_visible_explanation)
            return answer

        requested_target = plan.primary_prediction_target
        known_entities = set(self.adapter.policy_entity_references())
        execution_refusal_reasons: list[str] = []
        if frame_resolution_error is not None:
            execution_refusal_reasons.append(frame_resolution_error)
        if requested_target is None:
            execution_refusal_reasons.append(
                "QueryPlan must bind exactly one prediction_target before "
                "policy execution."
            )
            target = self.adapter.default_target_entity(snapshot)
        elif requested_target not in known_entities:
            execution_refusal_reasons.append(
                f"Unknown policy-controlled entity: {requested_target}."
            )
            target = self.adapter.default_target_entity(snapshot)
        else:
            target = requested_target
        if plan.clarification_required:
            execution_refusal_reasons.append(
                plan.clarification_reason
                or "The Transformer marked the request as ambiguous."
            )
        if plan.unsupported_components:
            execution_refusal_reasons.extend(plan.unsupported_components)
        if (
            plan.scene_edit_plan is not None
            and plan.scene_edit_plan.source_frame is not None
            and plan.scene_edit_plan.source_frame != snapshot.frame
        ):
            execution_refusal_reasons.append(
                "SceneEditPlan source frame does not match the restored checkpoint "
                f"({plan.scene_edit_plan.source_frame} != {snapshot.frame})."
            )
        baseline_outputs = self.adapter.policy_outputs(snapshot, self.policy)
        baseline_output = baseline_outputs[target]
        baseline_distribution = dict(
            zip(
                baseline_output.actions,
                baseline_output.probabilities,
            )
        )

        scene_edit: SceneEditResult | None = None
        edited_snapshot = snapshot
        if plan.requires_scene_edit and not execution_refusal_reasons:
            if plan.scene_edit_plan is None:
                raise ValueError("The query requires an edit but has no SceneEditPlan.")
            try:
                scene_edit = self.interventions.apply_plan(
                    snapshot,
                    plan.scene_edit_plan,
                )
                edited_snapshot = scene_edit.edited_snapshot
            except (TypeError, ValueError) as exc:
                execution_refusal_reasons.append(
                    f"Scene edit was rejected: {exc}"
                )

        edited_outputs = self.adapter.policy_outputs(
            edited_snapshot,
            self.policy,
        )
        edited_output = edited_outputs[target]
        edited_distribution = dict(
            zip(
                edited_output.actions,
                edited_output.probabilities,
            )
        )
        observation_changed = not np.array_equal(
            np.asarray(snapshot.observations[target]),
            np.asarray(edited_snapshot.observations[target]),
        )

        counterfactual: CounterfactualResult | None = None
        if (
            plan.requires_scene_edit
            or plan.requires_simulation
            or plan.requires_baseline_comparison
        ) and not execution_refusal_reasons:
            counterfactual = self.counterfactuals.simulate(
                snapshot,
                scene_edit.interventions if scene_edit else (),
                horizon=max(1, plan.horizon),
                repetitions=max(1, plan.rollout_count),
                deterministic=plan.rollout_count == 1,
                seed=seed,
            )

        why_analysis = None
        why_not_recourse = None
        direct_program_question = bool(
            self.program is not None
            and plan.requires_program_trace
            and not plan.requires_scene_edit
            and not plan.requires_simulation
            and not plan.requires_baseline_comparison
        )
        if (
            plan.intent == QueryIntent.EXPLANATORY
            and scene_edit is None
            and not direct_program_question
            and not execution_refusal_reasons
        ):
            why_analysis = self.counterfactuals.analyze_why(
                snapshot,
                target,
                repetitions=max(2, plan.rollout_count),
                seed=seed,
            )
            if not why_analysis.supported_candidates:
                execution_refusal_reasons.append(
                    "Current execution evidence is insufficient to identify a "
                    "tested observable factor with a significant policy effect."
                )
        elif (
            plan.intent == QueryIntent.WHY_NOT
            and not execution_refusal_reasons
        ):
            desired_action = _desired_action(
                plan,
                target,
                self.policy.action_names,
            )
            desired_objective = _desired_objective(
                plan,
                target,
                tuple(self.adapter.objective_descriptions()),
            )
            if desired_action is not None:
                why_not_recourse = self.counterfactuals.search_why_not(
                    snapshot,
                    target,
                    desired_action,
                    repetitions=max(2, plan.rollout_count),
                    seed=seed,
                )
                if not why_not_recourse.achieved:
                    execution_refusal_reasons.append(
                        why_not_recourse.refusal_reason
                        or "No tested legal recourse achieved the requested action."
                    )
            elif desired_objective is not None:
                # Objective selection is deterministic environment/task logic.
                # Its typed comparison evidence is already part of the shared
                # state bundle, so an action-level recourse rollout would answer
                # a different question and is intentionally not run.
                pass
            else:
                execution_refusal_reasons.append(
                    "Why-not execution requires a structured desired action or "
                    "objective in QueryPlan.desired_outcomes."
                )

        program_trace: tuple[Mapping[str, Any], ...] = ()
        program_limitation: str | None = None
        program_audit_note: str | None = None
        disagreement: dict[str, Any] = {
            "program_available": self.program is not None,
            "explanation_source": (
                "extracted_python_program"
                if self.program is not None
                else "unavailable"
            ),
        }
        # Build the semantic context once for the deployed Actor decision.
        # Both randomized conditions use these observable values.  The
        # extracted program is executed over the same context for audit only.
        semantic_context = self.adapter.semantic_policy_context(
            edited_snapshot,
            target,
        )
        if self.program is not None:
            features = semantic_context.features
            feature_descriptions = (
                self.adapter.semantic_feature_descriptions()
            )
            try:
                program_execution = self.program.execute(
                    features,
                    semantic_context,
                )
            except Exception as exc:
                # Program execution is an optional read-only audit.  A broken
                # export must never change the neural answer, simulation, or
                # refusal decision; represent it as an unavailable trace and
                # keep the failure exclusively in trace_audit.
                program_limitation = (
                    "The extracted program could not be executed for this "
                    "audit frame, so no program path was exposed."
                )
                program_audit_note = (
                    f"program_execution_failed:{type(exc).__name__}:"
                    f"{_brief_exception(exc)}"
                )
                fallback_action = max(
                    edited_distribution,
                    key=edited_distribution.__getitem__,
                )
                program_execution = SimpleNamespace(
                    action=fallback_action,
                    probabilities=dict(edited_distribution),
                    trace=SimpleNamespace(
                        tree_steps=(),
                        excluded_actions=(),
                        pre_mask_distribution=dict(edited_distribution),
                        regularization_version=False,
                    ),
                )
            program_distribution = dict(program_execution.probabilities)
            trace = program_execution.trace.tree_steps
            tree_trace_entries: list[Mapping[str, Any]] = []
            for index, step in enumerate(trace):
                bound_entities = dict(
                    self.adapter.semantic_feature_entity_bindings(
                        step.feature,
                        semantic_context,
                    )
                )
                observed_meaning = dict(
                    self.adapter.semantic_feature_observation(
                        step.feature,
                        step.observed_value,
                    )
                )
                if bound_entities:
                    observed_meaning["entity_bindings"] = bound_entities
                tree_trace_entries.append(
                    {
                        "branch_id": f"branch_{index + 1}",
                        "trace_type": "tree_branch",
                        "program_selected_action": (
                            program_execution.action
                        ),
                        **asdict(step),
                        "observed_relation": (
                            f"{step.observed_value} <= {step.threshold}"
                            if step.result
                            else f"{step.observed_value} > {step.threshold}"
                        ),
                        "feature_description": dict(
                            feature_descriptions.get(step.feature, {})
                        ),
                        "bound_entities": bound_entities,
                        "observed_meaning": observed_meaning,
                    }
                )
            tree_trace = tuple(tree_trace_entries)
            constraint_trace: list[Mapping[str, Any]] = []
            ordered_exclusions = sorted(
                program_execution.trace.excluded_actions,
                key=lambda item: (
                    -float(
                        features.get(
                            f"candidate.{item.action}.geometric_goal_progress",
                            0.0,
                        )
                    ),
                    -float(
                        program_execution.trace.pre_mask_distribution.get(
                            item.action,
                            0.0,
                        )
                    ),
                    item.action,
                ),
            )
            for index, exclusion in enumerate(ordered_exclusions):
                reason_observations = tuple(
                    {
                        "feature": reason_feature,
                        **dict(
                            self.adapter.semantic_feature_observation(
                                reason_feature,
                                float(features.get(reason_feature, 0.0)),
                            )
                        ),
                    }
                    for reason_feature in exclusion.active_reason_features
                )
                geometric_progress = float(
                    features.get(
                        f"candidate.{exclusion.action}.geometric_goal_progress",
                        0.0,
                    )
                )
                constraint_trace.append(
                    {
                        "branch_id": f"constraint_{index + 1}",
                        "trace_type": "action_constraint",
                        "action": program_execution.action,
                        "constrained_action": exclusion.action,
                        "legality_feature": exclusion.legality_feature,
                        "active_reason_features": exclusion.active_reason_features,
                        "bound_entities": dict(exclusion.bound_entities),
                        "geometric_goal_progress": geometric_progress,
                        "pre_mask_probability": float(
                            program_execution.trace.pre_mask_distribution.get(
                                exclusion.action,
                                0.0,
                            )
                        ),
                        "observed_meaning": {
                            "explanation_role": "action_feasibility",
                            "action": program_execution.action,
                            "constrained_action": exclusion.action,
                            "reason_observations": reason_observations,
                            "entity_bindings": dict(exclusion.bound_entities),
                            "explanation_requirements": (
                                {
                                    "key": (
                                        "action.constraint."
                                        f"{exclusion.action.lower()}"
                                    ),
                                    "semantic_name": "observable_action_constraint",
                                    "role": "action_reason",
                                    "group": "feasibility",
                                    "action": program_execution.action,
                                    "constrained_action": exclusion.action,
                                    "reason_features": exclusion.active_reason_features,
                                    "entity_bindings": dict(
                                        exclusion.bound_entities
                                    ),
                                },
                            ),
                        },
                    }
                )
            program_trace = (*tree_trace, *constraint_trace)
            disagreement.update(
                {
                    "neural_action": max(
                        edited_distribution,
                        key=edited_distribution.__getitem__,
                    ),
                    "program_action": max(
                        program_distribution,
                        key=program_distribution.__getitem__,
                    ),
                    # v2 programs expose typed execution paths and observable
                    # action constraints.  The explanation layer uses this
                    # flag admits only typed records the shared generator can
                    # actually ground in observable execution semantics.
                    "program_path_grounding": bool(
                        program_execution.trace.regularization_version
                        and _program_trace_is_groundable(program_trace)
                    ),
                    "program_distribution": program_distribution,
                    "l1_distribution_distance": sum(
                        abs(
                            float(edited_distribution.get(action, 0.0))
                            - float(program_distribution.get(action, 0.0))
                        )
                        for action in set(edited_distribution) | set(program_distribution)
                    ),
                    "program_complexity": self.program.complexity(),
                    "constraint_trace_available": bool(
                        getattr(self.program, "metadata", {}).get(
                            "action_constraint_reason_features"
                        )
                    ),
                }
            )
            local_program_reliability = _local_program_reliability(
                self.program,
                scenario_tags=semantic_context.scenario_tags,
                neural_action=str(disagreement["neural_action"]),
                program_action=str(disagreement["program_action"]),
            )
            disagreement["program_reliable"] = bool(
                local_program_reliability["reliable"]
            )
            disagreement["local_reliability"] = (
                local_program_reliability
            )
            if program_audit_note is not None:
                # Preserve the execution failure as the most specific reason.
                pass
            elif (
                disagreement["neural_action"]
                != disagreement["program_action"]
            ):
                program_limitation = (
                    "The extracted Python program chooses a different action "
                    "from the current neural policy at this frame, so its rule "
                    "path cannot be presented as the exact reason for the "
                    "neural action."
                )
            elif not disagreement["program_path_grounding"]:
                program_limitation = (
                    "The loaded program does not expose a typed execution path."
                )
            elif not disagreement["program_reliable"]:
                program_limitation = (
                    "The extracted Python program matches the action at this "
                    "frame but has not reached the required held-out local "
                    "fidelity for this kind of interaction. Its tree path is "
                    "therefore withheld from the explanation; observable "
                    "state and action-constraint evidence remain available."
                )
            elif not disagreement["constraint_trace_available"]:
                program_audit_note = (
                    "The loaded program predates relational action-constraint "
                    "traces; its tree path is available but detailed exclusion "
                    "reasons are not."
                )

        why_program_alignments: dict[str, Mapping[str, Any]] = {}
        direct_program_alignment: Mapping[str, Any] | None = None
        if self.program is not None:
            if why_analysis is not None:
                why_program_alignments = {
                    item.candidate.candidate_id: (
                        _program_counterfactual_alignment(
                            self.program,
                            self.adapter,
                            item.counterfactual,
                            target,
                        )
                    )
                    for item in why_analysis.candidates
                }
                if (
                    why_analysis.supported_candidates
                    and not any(
                        why_program_alignments.get(
                            item.candidate.candidate_id,
                            {},
                        ).get("direction_consistent", False)
                        for item in why_analysis.supported_candidates
                    )
                ):
                    disagreement["why_program_alignment_available"] = False
            if counterfactual is not None and scene_edit is not None:
                direct_program_alignment = (
                    _program_counterfactual_alignment(
                        self.program,
                        self.adapter,
                        counterfactual,
                        target,
                    )
                )
        direct_delta = (
            dict(
                counterfactual.action_probability_delta.get(target, {})
            )
            if counterfactual is not None
            else {}
        )
        direct_total_variation = 0.5 * sum(
            abs(float(value)) for value in direct_delta.values()
        )
        direct_causal_allowed = bool(
            scene_edit is not None
            and counterfactual is not None
            and observation_changed
            and direct_total_variation >= 0.02
        )
        if (
            plan.intent == QueryIntent.EXPLANATORY
            and scene_edit is not None
            and counterfactual is not None
            and not direct_causal_allowed
        ):
            execution_refusal_reasons.append(
                "The legal intervention did not produce sufficiently strong, "
                "observable evidence for the proposed "
                "causal explanation."
            )

        facts = self.adapter.evidence_facts(
            edited_snapshot,
            target,
            self.policy,
        )
        state_facts = tuple(
            {
                "fact_id": fact.fact_id,
                "predicate": fact.predicate,
                "arguments": fact.arguments,
                "value": fact.value,
                "factor_groups": fact.factor_groups,
                "verbalizations": fact.verbalizations,
                "value_verbalizations": fact.value_verbalizations,
            }
            for fact in facts
        )
        shared_task_context = _shared_task_context(
            state_facts,
            target=target,
        )
        baseline_results: tuple[Mapping[str, Any], ...] = ()
        counterfactual_results: tuple[Mapping[str, Any], ...] = ()
        if counterfactual is not None:
            baseline_results = tuple(
                {
                    "rollout_id": f"baseline_{index}",
                    "seed": pair.seed,
                    "first_action_distribution": (
                        _first_distributions(pair.baseline)
                    ),
                    "terminal_reason": pair.baseline.terminal_reason,
                    "action_frequencies": pair.baseline.action_frequencies,
                    "first_step": (
                        _frame_evidence(pair.baseline.frames[0])
                        if pair.baseline.frames
                        else None
                    ),
                    "total_reward": _rollout_total_reward(pair.baseline),
                }
                for index, pair in enumerate(counterfactual.paired_rollouts)
            )
            counterfactual_results = tuple(
                {
                    "rollout_id": f"counterfactual_{index}",
                    "seed": pair.seed,
                    "first_action_distribution": (
                        _first_distributions(pair.counterfactual)
                    ),
                    "action_probability_delta": dict(
                        counterfactual.action_probability_delta
                    ),
                    "terminal_reason": pair.counterfactual.terminal_reason,
                    "action_frequencies": (
                        pair.counterfactual.action_frequencies
                    ),
                    "total_reward": _rollout_total_reward(
                        pair.counterfactual
                    ),
                    "first_step": (
                        _frame_evidence(pair.counterfactual.frames[0])
                        if pair.counterfactual.frames
                        else None
                    ),
                }
                for index, pair in enumerate(counterfactual.paired_rollouts)
            )

        limitations: list[str] = list(execution_refusal_reasons)
        first_edited_frame = (
            counterfactual.paired_rollouts[0].counterfactual.frames[0]
            if counterfactual
            and counterfactual.paired_rollouts
            and counterfactual.paired_rollouts[0].counterfactual.frames
            else None
        )
        recorded_decision = bool(
            scene_edit is None
            and edited_snapshot.metadata.get(
                "decision_evidence_aligned",
                False,
            )
        )
        proposed_action = (
            first_edited_frame.proposed_actions.get(target)
            if first_edited_frame is not None
            else (
                edited_snapshot.proposed_actions.get(target)
                if recorded_decision
                else edited_output.proposed_action
            )
        )
        executed_action = (
            first_edited_frame.executed_actions.get(target)
            if first_edited_frame is not None
            else (
                edited_snapshot.executed_actions.get(target)
                if recorded_decision
                else proposed_action
            )
        )
        raw_action_resolution = (
            first_edited_frame.info.get("action_resolution", {})
            if first_edited_frame is not None
            else edited_snapshot.metadata.get("action_resolution", {})
            if recorded_decision
            else {}
        )
        action_resolution = (
            dict(raw_action_resolution.get(target, {}))
            if isinstance(raw_action_resolution, Mapping)
            and isinstance(raw_action_resolution.get(target), Mapping)
            else {}
        )
        resolved_action = str(
            action_resolution.get("resolved_action", proposed_action or "")
        )
        environment_changed_action = bool(
            proposed_action is not None
            and executed_action is not None
            and proposed_action != executed_action
        )
        edited_argmax_action = max(
            edited_distribution,
            key=edited_distribution.__getitem__,
        )
        requested_alternative_action = _desired_action(
            plan,
            target,
            self.policy.action_names,
        )
        observable_decision_context = tuple(
            self.adapter.neural_baseline_explanation_context(
                semantic_context,
                str(proposed_action or edited_argmax_action),
            )
        )
        # Ordinary Why explanations only need the selected action's Actor-visible
        # context.  A Why-not question additionally names one rejected action;
        # preserve that action's context in the shared bundle so both
        # experimental conditions can answer the requested contrast directly.
        # This is observation-derived evidence, never an extracted-program trace.
        why_not_action_context = (
            tuple(
                {
                    **dict(item),
                    "queried_action": requested_alternative_action,
                }
                for item in self.adapter.neural_baseline_explanation_context(
                    semantic_context,
                    requested_alternative_action,
                )
            )
            if requested_alternative_action is not None
            else ()
        )
        # One authoritative execution context is shared by both explanation
        # conditions.  Program evidence can explain the Actor proposal; these
        # transition records explain any different final joint action.
        execution_reason_context = tuple(
            fact
            for fact in state_facts
            if (
                {"action", "action_reason"}.intersection(
                    {str(group) for group in fact.get("factor_groups", ())}
                )
                and (
                    not fact.get("arguments")
                    or target
                    in {str(value) for value in fact.get("arguments", ())}
                )
            )
        )
        decision_causal_record = _decision_causal_record(
            target=target,
            argmax_action=edited_argmax_action,
            proposed_action=proposed_action,
            resolved_action=resolved_action,
            executed_action=executed_action,
            action_resolution=action_resolution,
            observable_decision_context=observable_decision_context,
            execution_reason_context=execution_reason_context,
        )
        action_descriptions = {
            str(action): dict(descriptions)
            for action, descriptions in (
                self.adapter.action_descriptions().items()
            )
        }
        objective_descriptions = {
            str(objective): dict(descriptions)
            for objective, descriptions in (
                self.adapter.objective_descriptions().items()
            )
        }
        question_vocabulary = dict(
            self.adapter.question_vocabulary()
            if hasattr(self.adapter, "question_vocabulary")
            else {}
        )
        supported_causal = bool(
            why_analysis
            and any(
                item.supported
                and item.observable_by_actor
                for item in why_analysis.candidates
            )
        )
        counterfactual_contrast = _counterfactual_decision_contrast(
            self.adapter,
            self.policy,
            original_snapshot=snapshot,
            edited_snapshot=edited_snapshot,
            target=target,
            original_distribution=baseline_distribution,
            edited_distribution=edited_distribution,
            scene_edit=scene_edit,
            counterfactual=counterfactual,
            program=None,
            program_alignment=None,
        )
        common_evidence = EvidenceBundle(
            query_plan=plan,
            direct_result={
                "execution_refused": bool(execution_refusal_reasons),
                "refusal_reasons": tuple(execution_refusal_reasons),
                "executed_frame": snapshot.frame,
                "decision_outcome_frame": (
                    snapshot.metadata.get(
                        "decision_outcome_frame"
                    )
                ),
                "target": target,
                "baseline_action_distribution": baseline_distribution,
                "edited_action_distribution": edited_distribution,
                "edited_argmax_action": edited_argmax_action,
                "argmax_action": edited_argmax_action,
                "recorded_proposed_action": proposed_action,
                "recorded_executed_action": executed_action,
                "proposed_action": proposed_action,
                "resolved_action": resolved_action,
                "executed_action": executed_action,
                "environment_changed_action": environment_changed_action,
                "action_resolution": action_resolution,
                "action_descriptions": action_descriptions,
                "objective_descriptions": objective_descriptions,
                "question_vocabulary": question_vocabulary,
                "baseline_actor_output": distribution_evidence(
                    baseline_output
                ),
                "edited_actor_output": distribution_evidence(
                    edited_output
                ),
                "counterfactual_contrast": counterfactual_contrast,
                "shared_task_context": shared_task_context,
                "observable_decision_context": observable_decision_context,
                "why_not_action_context": why_not_action_context,
                "execution_reason_context": execution_reason_context,
                "decision_causal_record": decision_causal_record,
            },
            state_facts=state_facts,
            interventions=tuple(
                asdict(item) for item in (scene_edit.interventions if scene_edit else ())
            ),
            baseline_results=baseline_results,
            counterfactual_results=counterfactual_results,
            policy_results={
                "target": target,
                # Architectural invariant: the program may contribute a local
                # audit trace, but it is never installed as the controller or
                # as the policy used for a user-requested rollout.
                "controller_policy": "neural_policy",
                "baseline_distribution": baseline_distribution,
                "edited_distribution": edited_distribution,
                "all_agent_actor_outputs": {
                    agent_id: distribution_evidence(output)
                    for agent_id, output in edited_outputs.items()
                },
                "causal_observable": bool(
                    observation_changed or supported_causal
                ),
                "causal_claim_allowed": bool(
                    supported_causal or direct_causal_allowed
                ),
                "action_mask": tuple(edited_output.action_mask),
                "argmax_action": edited_argmax_action,
                "proposed_action": proposed_action,
                "resolved_action": resolved_action,
                "executed_action": executed_action,
                "action_descriptions": action_descriptions,
                "objective_descriptions": objective_descriptions,
                "question_vocabulary": question_vocabulary,
                "proposed_action_probability": (
                    float(
                        edited_distribution.get(
                            proposed_action,
                            0.0,
                        )
                    )
                    if proposed_action is not None
                    else None
                ),
                "selection_mode": (
                    "deterministic_argmax"
                    if bool(
                        edited_snapshot.metadata.get(
                            "decision_deterministic",
                            False,
                        )
                    )
                    else "recorded_stochastic_sample"
                    if recorded_decision
                    else "policy_query_argmax"
                ),
                "environment_enforced_action_change": (
                    environment_changed_action
                ),
                "environment_changed_action": environment_changed_action,
                "environment_action_resolution": action_resolution,
                "action_resolution": action_resolution,
                "observable_decision_context": observable_decision_context,
                "why_not_action_context": why_not_action_context,
                "execution_reason_context": execution_reason_context,
                "decision_causal_record": decision_causal_record,
                "shared_task_context": shared_task_context,
                "counterfactual_contrast": counterfactual_contrast,
            },
            program_trace=(),
            disagreement={},
            causal_analysis=(
                _why_analysis_evidence(
                    why_analysis,
                    {},
                )
                if why_analysis is not None
                else {
                    "supported": direct_causal_allowed,
                    "direct_intervention": bool(scene_edit is not None),
                    "actor_observation_changed": observation_changed,
                    "neural_total_variation": direct_total_variation,
                    "neural_probability_delta": direct_delta,
                    "causal_claim_allowed": direct_causal_allowed,
                }
                if counterfactual is not None
                else {}
            ),
            why_not_recourse=(
                _why_not_evidence(why_not_recourse)
                if why_not_recourse is not None
                else {}
            ),
            uncertainty={
                "paired_rollout_count": (
                    len(counterfactual.paired_rollouts) if counterfactual else 0
                ),
                "stochastic": bool(plan.rollout_count > 1),
            },
            limitations=tuple(limitations),
        )
        trace_available = bool(program_trace)
        trace_eligible = bool(
            self.program is not None
            and trace_available
            and disagreement.get("program_path_grounding", False)
            and disagreement.get("program_reliable", False)
            and disagreement.get("neural_action")
            == disagreement.get("program_action")
        )
        selected_trace = (
            program_trace
            if (
                include_program_trace
                and trace_eligible
                and plan.requires_program_trace
            )
            else ()
        )
        evidence = replace(
            common_evidence,
            program_trace=selected_trace,
            # Reliability, action agreement, and simulation alignment are
            # audit/gating variables, never additional prompt facts.
            disagreement={},
        )
        trace_audit: dict[str, Any] = {
            "assigned_condition": condition.value,
            "available": trace_available,
            "eligible": trace_eligible,
            "exposed": False,
            "semantic_plan_admitted": False,
            "raw_candidate_exposed": False,
            "display_program_evidence_ids": (),
            "withheld_reason": (
                None
                if selected_trace
                else "assigned_no_trace_condition"
                if not include_program_trace
                else "question_does_not_request_action_trace"
                if trace_eligible and not plan.requires_program_trace
                else program_limitation
                or (
                    "No extracted program was loaded."
                    if self.program is None
                    else "No explanation-bearing execution path was available."
                )
            ),
            "audit_note": program_audit_note,
            "action_agreement": bool(
                disagreement.get("neural_action")
                == disagreement.get("program_action")
            )
            if self.program is not None
            else False,
            "local_reliability": dict(
                disagreement.get("local_reliability", {})
            ),
            "policy_sha256": self.policy_artifact_hash,
            "program_sha256": self.program_artifact_hash,
            "program_counterfactual_alignment": dict(
                direct_program_alignment or {}
            ),
            "why_program_alignments": {
                str(key): dict(value)
                for key, value in why_program_alignments.items()
            },
            "fallback_status": "pending",
        }
        requested_language = (
            plan.response_language
            if language.lower() in {"auto", "und"}
            else language
        )
        evidence_ready_at = time.perf_counter()
        ir_started = time.perf_counter()
        common_ir_key = _common_ir_cache_key(
            evidence,
            language=requested_language,
            policy_hash=self.policy_artifact_hash,
            program_hash=self.program_artifact_hash,
            environment_id=(
                f"{type(self.adapter).__module__}."
                f"{type(self.adapter).__qualname__}"
            ),
        )
        ir_cache_hit = common_ir_key in self._ir_cache
        if ir_cache_hit:
            common_ir = self._ir_cache.pop(common_ir_key)
            self._ir_cache[common_ir_key] = common_ir
        else:
            common_ir = self.explanation_generator.compiler.compile_common(
                evidence,
                requested_language=requested_language,
                semantics=self.explanation_generator.semantics,
            )
            self._ir_cache[common_ir_key] = common_ir
            while len(self._ir_cache) > self._ir_cache_size:
                self._ir_cache.popitem(last=False)
        explanation_ir = (
            self.explanation_generator.compiler.add_trace(
                common_ir,
                evidence,
                semantics=self.explanation_generator.semantics,
            )
            if selected_trace
            else common_ir
        )
        ir_compile_ms = (time.perf_counter() - ir_started) * 1000.0
        trace_audit["semantic_plan_admitted"] = bool(explanation_ir.trace_units)
        cache_key = _document_cache_key(
            explanation_ir.ir_hash,
            policy_hash=self.policy_artifact_hash,
            program_hash=self.program_artifact_hash,
            environment_id=(
                f"{type(self.adapter).__module__}."
                f"{type(self.adapter).__qualname__}"
            ),
        )
        realization_started = time.perf_counter()
        cache_hit = cache_key in self._document_cache
        if cache_hit:
            cached_realization = self._document_cache.pop(cache_key)
            self._document_cache[cache_key] = cached_realization
            document = cached_realization.document
            raw_document = cached_realization.raw_document
            raw_candidate_text = cached_realization.raw_text
            realization_model_calls = 0
            self.explanation_generator.last_generation_metrics = {}
            generation_grounding = (
                self.explanation_generator.grounding_for_document(
                    explanation_ir,
                    document,
                    model_call_count=0,
                )
            )
            generation_grounding["raw_candidate"] = {
                "text": raw_candidate_text,
                "document": (
                    raw_document.to_dict()
                    if raw_document is not None
                    else None
                ),
                "used_unit_ids": (
                    raw_document.used_unit_ids
                    if raw_document is not None
                    else ()
                ),
            }
        else:
            document = self.explanation_generator.generate_document(
                evidence,
                include_program_trace=bool(selected_trace),
                language=requested_language,
                explanation_ir=explanation_ir,
            )
            realization_model_calls = int(
                self.explanation_generator.last_model_call_count
            )
            generation_grounding = dict(
                self.explanation_generator.last_grounding
            )
            raw_document = self.explanation_generator.last_raw_document
            raw_candidate_text = (
                self.explanation_generator.last_raw_candidate_text
            )
            self._document_cache[cache_key] = _CachedRealization(
                document=document,
                raw_document=raw_document,
                raw_text=raw_candidate_text,
            )
            while len(self._document_cache) > self._document_cache_size:
                self._document_cache.popitem(last=False)
        realization_ms = (time.perf_counter() - realization_started) * 1000.0
        validation_started = time.perf_counter()
        final_validation_issues = validate_document(document, explanation_ir)
        self._document_cache[cache_key] = _CachedRealization(
            document=document,
            raw_document=raw_document,
            raw_text=raw_candidate_text,
        )
        display_explanation = document.text
        claims, verdicts = _claims_and_verdicts_from_document(
            explanation_ir,
            document,
            frame=plan.frame_reference,
        )
        validation_ms = (time.perf_counter() - validation_started) * 1000.0
        display_claims = claims
        display_verdicts = verdicts
        raw_claims, raw_verdicts = (
            _claims_and_verdicts_from_document(
                explanation_ir,
                raw_document,
                frame=plan.frame_reference,
            )
            if raw_document is not None
            else ((), ())
        )
        posthoc_warnings = list(final_validation_issues)
        display_program_evidence_ids = tuple(
            dict.fromkeys(
                evidence_id
                for unit in explanation_ir.trace_units
                if unit.unit_id in document.used_unit_ids
                for evidence_id in unit.evidence_ids
            )
        )
        trace_audit["raw_candidate_exposed"] = bool(
            selected_trace and display_program_evidence_ids
        )
        trace_audit["display_program_evidence_ids"] = display_program_evidence_ids
        trace_audit["exposed"] = bool(
            selected_trace and display_program_evidence_ids
        )
        if trace_audit["exposed"]:
            trace_audit["withheld_reason"] = None
        elif selected_trace and trace_audit["semantic_plan_admitted"]:
            trace_audit["withheld_reason"] = "display_text_did_not_realize_execution_trace"
        elif selected_trace:
            trace_audit["withheld_reason"] = "semantic_plan_did_not_admit_execution_trace"
        display_gate_status = "deterministic_conversational"
        trace_audit["fallback_status"] = display_gate_status
        planner_diagnostics = dict(_parse_diagnostics or {})
        generation_diagnostics = {
            "schema_version": "generation-diagnostics.v3",
            "parse_ms": float(planner_diagnostics.get("latency_ms", 0.0)),
            "simulation_ms": (evidence_ready_at - execution_started) * 1000.0,
            "ir_compile_ms": ir_compile_ms,
            "realization_ms": realization_ms,
            "validation_ms": validation_ms,
            "total_ms": (
                (time.perf_counter() - execution_started) * 1000.0
                + float(planner_diagnostics.get("latency_ms", 0.0))
            ),
            "model_call_count": int(
                planner_diagnostics.get("model_call_count", 0)
            ) + realization_model_calls,
            "question_ir_model_calls": int(
                planner_diagnostics.get("model_call_count", 0)
            ),
            "realization_model_calls": 0,
            "question_cache_hit": bool(
                planner_diagnostics.get("cache_hit", False)
            ),
            "ir_cache_hit": ir_cache_hit,
            "document_cache_hit": cache_hit,
            "answer_cache_hit": False,
            "renderer": document.renderer,
            "parse_input_tokens": int(
                planner_diagnostics.get("input_tokens", 0)
            ),
            "parse_output_tokens": int(
                planner_diagnostics.get("output_tokens", 0)
            ),
            "realization_input_tokens": int(
                generation_grounding.get("input_tokens", 0)
            ),
            "realization_output_tokens": int(
                generation_grounding.get("output_tokens", 0)
            ),
            "question_binding": dict(
                planner_diagnostics.get("binding", {})
            ),
        }
        generation_grounding = {
            **generation_grounding,
            "posthoc_display_gate": {
                "status": display_gate_status,
                "raw_claim_count": len(claims),
                "supported_claim_count": len(verdicts),
                "withheld_claim_count": 0,
            },
        }
        if on_explanation is not None:
            on_explanation(display_explanation)
        answer = QueryAnswer(
            query_plan=plan,
            evidence=evidence,
            # Backward-compatible public fields now always contain the exact
            # participant-facing, post-gate result.  The candidate and its
            # audit remain available through the explicit raw_* fields.
            explanation=display_explanation,
            claims=tuple(display_claims or ()),
            verdicts=tuple(display_verdicts or ()),
            explanation_mode=condition.value,
            generation_grounding=generation_grounding,
            scene_edit=scene_edit,
            counterfactual=counterfactual,
            posthoc_warnings=tuple(posthoc_warnings),
            display_explanation=display_explanation,
            display_claims=display_claims,
            display_verdicts=display_verdicts,
            raw_explanation=(
                raw_candidate_text
                if raw_candidate_text is not None
                else display_explanation
            ),
            raw_claims=raw_claims,
            raw_verdicts=raw_verdicts,
            trace_audit=trace_audit,
            explanation_document=document.to_dict(),
            explanation_ir_hash=explanation_ir.ir_hash,
            generation_diagnostics=generation_diagnostics,
        )
        self._answer_cache[answer_cache_key] = answer
        while len(self._answer_cache) > self._answer_cache_size:
            self._answer_cache.popitem(last=False)
        return answer


from backend.simulation.query_evidence import (
    _answer_cache_key,
    _brief_exception,
    _cache_value,
    _claims_and_verdicts_from_document,
    _common_ir_cache_key,
    _counterfactual_decision_contrast,
    _decision_causal_record,
    _desired_action,
    _desired_objective,
    distribution_evidence,
    _document_cache_key,
    _first_distributions,
    _frame_evidence,
    _local_program_reliability,
    _model_cache_identity,
    _program_counterfactual_alignment,
    _program_trace_is_groundable,
    _shared_task_context,
    _why_analysis_evidence,
    _why_not_evidence,
)
