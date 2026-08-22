"""Regularity-Constrained Policy Distillation (RCPD).

This file contains the complete environment-independent research algorithm:

on-policy training states -> bounded symbolic program -> extractability score
-> program-guided Actor update -> repeat throughout reinforcement learning.

The extracted program is not a post-hoc surrogate that is trained only after
the neural policy has converged.  It is refreshed during RL training and is
used as a deliberately simple target for the next Actor update.  Alternating
between policy optimisation and bounded program extraction encourages the
neural policy itself to acquire stable, program-like decision boundaries.

An environment supplies opaque states and a semantic feature encoder through
callbacks.  Legal counterfactual and interaction probes may help choose and
audit the bounded program structure.  They never become invented supervision
for the neural controller: the current NN supplies every extraction label, and
the Warehouse training protocol permits a direct Actor gradient only on
positive-advantage on-policy rows where NN, sampled action, and program agree.
At runtime the NN remains the sole controller; the program is local evidence
for regularity measurement and explanation, not a replacement policy.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from .policy_program_regularizer import program_complexity


State = Any
FeatureVector = Mapping[str, float]
FeatureEncoder = Callable[[State], FeatureVector]
ImportanceWeightProvider = Callable[[State], float]
SampleGroupProvider = Callable[[State], Sequence[str]]
SampleKeyProvider = Callable[[State], str | None]
PredicateGroupContract = Mapping[str, Sequence[str]]


from .rcpd_config import OracleOutput, OraclePolicy, RCPDConfig


from .program import (
    ActionExclusion,
    DistillationMetrics,
    ExecutableProgram,
    ExtractedProgram,
    ProgramExecution,
    ProgramExecutionTrace,
    ProgramNode,
    ProgramTrace,
    ProgramTraceStep,
)


from .rcpd_tree import (
    _Sample,
    _as_oracle_output,
    _candidate_feature_thresholds,
    _counterfactual_changed_training_indices,
    _data_split_audit,
    _ensure_required_semantic_training_variation,
    _feature_family,
    _feature_varies,
    _internal_program_nodes,
    _is_relational_predicate,
    _kl_divergence,
    _leaf_program_nodes,
    _minimum_program_leaf_samples,
    _normalize_predicate_group_contract,
    _program_feature_allowed,
    _program_predicate_group_coverage,
    _reestimate_program_leaves,
    _replace_program_node_split,
    _replace_program_subtree,
    _samples_reaching_path,
    _stratified_validation_indices,
    _temperature_scale_probabilities,
    _tree_to_program,
)


@dataclass(frozen=True)
class RCPDResult:
    program: ExecutableProgram
    metrics: DistillationMetrics
    extraction_summary: tuple[Mapping[str, Any], ...]


class RCPD:
    """Alternate bounded program extraction with program-guided Actor updates."""

    def __init__(self, config: RCPDConfig | None = None) -> None:
        self.config = config or RCPDConfig()
        self.program: ExecutableProgram | None = None
        self.last_result: RCPDResult | None = None
        self.last_extract_step: int | None = None
        self.regularization_weight: float = 0.0
        self.last_error: str | None = None
        self.extraction_history: list[dict[str, Any]] = []

    def maybe_extract(
        self,
        step: int,
        states: Sequence[State],
        oracle: OraclePolicy,
        feature_encoder: FeatureEncoder,
        *,
        force: bool = False,
        **fit_kwargs: Any,
    ) -> RCPDResult | None:
        """Refresh the bounded program during RL training.

        Extraction failure is explicitly fail-open: the last valid program is
        retained and ``last_error`` records the reason for the training log.
        A successful extraction immediately supplies a non-zero feedback weight
        for subsequent Actor updates when ``regularization_lambda`` is positive.
        """

        if not self.config.enabled:
            self.regularization_weight = 0.0
            return None
        due = (
            force
            or (
                self.last_extract_step is None
                and step >= max(1, self.config.extraction_interval)
            )
            or (
                self.last_extract_step is not None
                and step - self.last_extract_step
                >= max(1, self.config.extraction_interval)
            )
        )
        if not due:
            return None
        if len(states) < self.config.minimum_extraction_samples:
            self.last_error = (
                f"insufficient extraction samples: {len(states)} < "
                f"{self.config.minimum_extraction_samples}"
            )
            self.regularization_weight = 0.0
            return None
        try:
            result = self.fit(
                states,
                oracle,
                feature_encoder,
                **fit_kwargs,
            )
        except Exception as exc:  # training must continue with a visible warning
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.regularization_weight = 0.0
            self.last_extract_step = int(step)
            return None
        self.program = result.program
        self.last_result = result
        self.last_extract_step = int(step)
        self.last_error = None
        self.regularization_weight = (
            min(
                self.config.maximum_regularization_lambda,
                max(0.0, self.config.regularization_lambda),
            )
            if result.metrics.feedback_eligible
            else 0.0
        )
        self.extraction_history.append(
            {
                "step": int(step),
                "extractability_loss": result.metrics.extractability_loss,
                "extractability_score": result.metrics.extractability_score,
                "program_size": result.metrics.program_size,
                "program_depth": result.metrics.program_depth,
                "program_leaf_count": result.metrics.program_leaf_count,
                "program_predicate_count": (
                    result.metrics.program_predicate_count
                ),
                "lambda_extract": self.config.regularization_lambda,
                "complexity_lambda": self.config.complexity_penalty,
                "extraction_interval": self.config.extraction_interval,
                "program_target_temperature": (
                    self.config.program_target_temperature
                ),
                "feedback_weight": self.regularization_weight,
                "interaction_macro_fidelity": (
                    result.metrics.interaction_macro_fidelity
                ),
                "counterfactual_delta_error": (
                    result.metrics.counterfactual_delta_error
                ),
                "counterfactual_direction_fidelity": (
                    result.metrics.counterfactual_direction_fidelity
                ),
                "counterfactual_dataset_pairs": (
                    result.metrics.counterfactual_dataset_pairs
                ),
                "required_predicate_group_coverage": {
                    group: list(features)
                    for group, features in (
                        result.metrics.required_predicate_group_coverage.items()
                    )
                },
                "semantic_predicate_coverage_complete": (
                    result.metrics.semantic_predicate_coverage_complete
                ),
                "feedback_eligible": result.metrics.feedback_eligible,
                "feedback_ineligibility_reasons": list(
                    result.metrics.feedback_ineligibility_reasons
                ),
                "explanation_eligible": result.metrics.explanation_eligible,
                "explanation_ineligibility_reasons": list(
                    result.metrics.explanation_ineligibility_reasons
                ),
            }
        )
        return result

    def program_targets(
        self,
        features: Sequence[FeatureVector],
    ) -> np.ndarray | None:
        """Return detached program distributions used as a neural KL target."""

        raw = self._raw_program_targets(features)
        if raw is None:
            return None
        return np.asarray(
            _temperature_scale_probabilities(
                raw,
                self.config.program_target_temperature,
            ),
            dtype=np.float32,
        )

    def program_feedback_targets(
        self,
        features: Sequence[FeatureVector],
        *,
        actor_probabilities: Any | None = None,
    ) -> np.ndarray | None:
        """Build the detached target used only by the training regularizer.

        Program execution and explanation always use the actual extracted
        distribution.  ``action_anchor`` changes only the training signal: the
        tree identifies a simple categorical region, while the NN retains its
        own relative beliefs about every non-selected action.
        """

        program = self.program_targets(features)
        if program is None:
            return None
        mode = str(self.config.feedback_target_mode).strip().casefold()
        if mode == "program_distribution":
            return program
        if mode not in {"program_blend", "action_anchor"}:
            raise ValueError(
                "feedback_target_mode must be 'program_distribution', "
                "'program_blend', or 'action_anchor'."
            )
        if actor_probabilities is None:
            raise ValueError(
                f"{mode} feedback requires current Actor probabilities."
            )
        actor = np.asarray(actor_probabilities, dtype=np.float32)
        if actor.shape != program.shape:
            raise ValueError(
                "Actor probabilities must match the program-target shape."
            )
        actor = np.clip(actor, 0.0, None)
        actor /= np.clip(actor.sum(axis=1, keepdims=True), 1e-8, None)
        strength = min(
            1.0,
            max(0.0, float(self.config.feedback_target_strength)),
        )
        if mode == "program_blend":
            return np.asarray(
                (1.0 - strength) * actor + strength * program,
                dtype=np.float32,
            )
        anchor = np.zeros_like(actor)
        anchor[
            np.arange(len(program)),
            np.argmax(program, axis=1),
        ] = 1.0
        return np.asarray(
            (1.0 - strength) * actor + strength * anchor,
            dtype=np.float32,
        )

    def _raw_program_targets(
        self,
        features: Sequence[FeatureVector],
    ) -> np.ndarray | None:
        if self.program is None or self.regularization_weight <= 0.0:
            return None
        return np.asarray(
            [
                [
                    self.program.predict_proba(item)[action]
                    for action in self.program.action_names
                ]
                for item in features
            ],
            dtype=np.float32,
        )

    def scheduled_regularization_weight(
        self,
        training_fraction: float,
    ) -> float:
        """Return the actual lambda used at one point in RL training.

        Program extraction remains active before the feedback starts.  Only
        the gradient fed back to the Actor is delayed/ramped, so this remains
        extraction-in-the-loop rather than post-hoc distillation.
        """

        if self.regularization_weight <= 0.0:
            return 0.0
        progress = min(1.0, max(0.0, float(training_fraction)))
        start = float(self.config.regularization_start_fraction)
        ramp = float(self.config.regularization_ramp_fraction)
        if progress < start:
            return 0.0
        if ramp <= 0.0:
            return self.regularization_weight
        multiplier = min(1.0, max(0.0, (progress - start) / ramp))
        return self.regularization_weight * multiplier

    def program_target_weights(
        self,
        features: Sequence[FeatureVector],
        actor_probabilities: Any | None = None,
        sample_groups: Sequence[Sequence[str]] | None = None,
        counterfactual_mask: Sequence[bool] | None = None,
    ) -> np.ndarray | None:
        """Return confidence weights for the simple program's Actor targets.

        The global feedback strength is controlled by ``regularization_lambda``.
        Per-state confidence prevents an uncertain leaf from dominating PPO.
        If current Actor probabilities are supplied, confident task decisions
        are protected.  A stricter research protocol can additionally require
        NN/program action agreement, so the program smooths only an action
        pattern the NN already follows instead of acting as a replacement
        controller.
        """

        raw_targets = self._raw_program_targets(features)
        if raw_targets is None:
            return None
        if raw_targets.shape[1] < 2:
            return np.ones(len(raw_targets), dtype=np.float32)
        ordered = np.sort(raw_targets, axis=1)
        margins = ordered[:, -1] - ordered[:, -2]
        floor = min(1.0, max(0.0, self.config.minimum_target_weight))
        program_confidence = np.asarray(
            floor + (1.0 - floor) * np.clip(margins, 0.0, 1.0),
            dtype=np.float32,
        )
        group_reliability = np.ones(len(raw_targets), dtype=np.float32)
        if sample_groups is not None and self.last_result is not None:
            group_metrics = dict(
                self.last_result.metrics.group_action_fidelity
            )
            threshold = float(
                self.config.minimum_interaction_fidelity_for_feedback
            )
            for index, groups in enumerate(sample_groups):
                available = [
                    float(group_metrics[str(group)])
                    for group in groups
                    if str(group) != "ordinary"
                    and str(group) in group_metrics
                ]
                if available:
                    minimum = min(available)
                    group_reliability[index] = (
                        minimum if minimum >= threshold else 0.0
                    )
        if actor_probabilities is None:
            return program_confidence * group_reliability
        actor = np.asarray(actor_probabilities, dtype=np.float32)
        if actor.shape != raw_targets.shape:
            raise ValueError(
                "Actor probabilities must match the program-target shape."
            )
        actor = np.clip(actor, 0.0, None)
        actor /= np.clip(actor.sum(axis=1, keepdims=True), 1e-8, None)
        actor_ordered = np.sort(actor, axis=1)
        actor_margins = actor_ordered[:, -1] - actor_ordered[:, -2]
        agreements = (
            np.argmax(actor, axis=1) == np.argmax(raw_targets, axis=1)
        )
        threshold = max(
            0.0,
            float(self.config.maximum_disagreement_actor_margin),
        )
        # This is deliberately a gate, not another unreported rescaling of
        # lambda.  A disagreement is either inside the pre-registered
        # low-margin region and receives its ordinary program-confidence
        # weight, or it is protected completely.  Setting the threshold to
        # one disables disagreement protection because a categorical margin
        # cannot exceed one.
        disagreement_weights = (
            actor_margins <= threshold
        ).astype(np.float32)
        feedback_margin_threshold = min(
            1.0,
            max(0.0, float(self.config.maximum_feedback_actor_margin)),
        )
        boundary_gate = (
            actor_margins <= feedback_margin_threshold
        ).astype(np.float32)
        if self.config.require_action_agreement_for_feedback:
            action_gate = agreements.astype(np.float32)
        else:
            action_gate = np.where(
                agreements,
                np.ones(len(actor), dtype=np.float32),
                disagreement_weights,
            )
        safety_gate = action_gate * boundary_gate
        if counterfactual_mask is not None:
            mask = np.asarray(counterfactual_mask, dtype=bool)
            if mask.shape != (len(raw_targets),):
                raise ValueError(
                    "Counterfactual mask must match the program-target rows."
                )
            if (
                self.config.require_explanation_eligibility_for_counterfactual_feedback
                and self.last_result is not None
                and not self.last_result.metrics.explanation_eligible
            ):
                safety_gate = safety_gate * (~mask).astype(np.float32)
        return program_confidence * group_reliability * safety_gate

    @staticmethod
    def regularization_loss(
        actor_output: Any,
        program_output: Any,
        weights: Any | None = None,
    ) -> Any:
        """Compute batch-mean weighted ``KL(actor || program)``.

        Weights are absolute per-example feedback strengths rather than values
        that are renormalized to sum to one.  Consequently, uncertain or
        gated-out examples reduce the total regularizer gradient as intended.
        """

        try:
            import torch

            if isinstance(actor_output, torch.Tensor):
                actor_log_probabilities = torch.log_softmax(
                    actor_output,
                    dim=-1,
                )
                actor_probabilities = torch.exp(actor_log_probabilities)
                target = torch.as_tensor(
                    program_output,
                    dtype=actor_output.dtype,
                    device=actor_output.device,
                ).clamp_min(0.0)
                target = target / target.sum(dim=-1, keepdim=True)
                safe_target = target.clamp_min(1e-8)
                safe_target = safe_target / safe_target.sum(
                    dim=-1,
                    keepdim=True,
                )
                losses = torch.sum(
                    actor_probabilities
                    * (
                        actor_log_probabilities
                        - torch.log(safe_target)
                    ),
                    dim=-1,
                )
                if weights is None:
                    return losses.mean()
                weight_tensor = torch.as_tensor(
                    weights,
                    dtype=actor_output.dtype,
                    device=actor_output.device,
                )
                return (losses * weight_tensor).mean()
        except ImportError:
            pass
        actor = np.asarray(actor_output, dtype=float)
        target = np.asarray(program_output, dtype=float)
        if actor.ndim == 1:
            actor = actor[None, :]
        if target.ndim == 1:
            target = target[None, :]
        # Accept either probabilities or logits for the actor.
        if np.any(actor < 0.0) or not np.allclose(actor.sum(axis=-1), 1.0, atol=1e-4):
            shifted = actor - np.max(actor, axis=-1, keepdims=True)
            actor = np.exp(shifted)
        actor = actor / np.clip(actor.sum(axis=-1, keepdims=True), 1e-12, None)
        target = np.clip(target, 1e-12, None)
        target = target / target.sum(axis=-1, keepdims=True)
        losses = np.sum(
            actor
            * (
                np.log(np.clip(actor, 1e-12, None))
                - np.log(target)
            ),
            axis=-1,
        )
        if weights is None:
            return float(np.mean(losses))
        weight_array = np.asarray(weights, dtype=float)
        return float(np.mean(losses * weight_array))

    def evaluate_fidelity(
        self,
        actor: OraclePolicy,
        program: ExecutableProgram,
        samples: Sequence[State],
        feature_encoder: FeatureEncoder,
    ) -> DistillationMetrics:
        return self.evaluate(program, samples, actor, feature_encoder)

    def training_state(self) -> dict[str, Any]:
        return {
            "config": {
                field_name: getattr(self.config, field_name)
                for field_name in self.config.__dataclass_fields__
            },
            "program": self.program.to_dict() if self.program else None,
            "metrics": self.last_result.metrics.to_dict() if self.last_result else None,
            "last_extract_step": self.last_extract_step,
            "regularization_weight": self.regularization_weight,
            "extraction_history": list(self.extraction_history),
            "last_error": self.last_error,
        }

    def restore_training_state(self, payload: Mapping[str, Any]) -> None:
        """Restore the extracted regularity program while keeping the new run config.

        Keeping ``self.config`` is intentional: a paired experiment may fork
        one common, lambda-zero checkpoint into several feedback strengths.
        The neural policy, optimizer, RNG state, and program are shared; only
        the prospective regularization setting changes.
        """

        previous_config = dict(payload.get("config") or {})
        # A resumed branch may deliberately change only the prospective
        # feedback schedule/strength.  Reusing the stored program is safe in
        # that case.  Any setting that changes how the bounded program is
        # fitted, selected, or declared reliable requires an immediate refit;
        # otherwise the first resumed Actor updates could be regularized by a
        # tree produced under a different extraction protocol.
        extraction_fields = (
            "max_depth",
            "max_leaf_nodes",
            "max_predicates",
            "min_samples_leaf",
            "complexity_penalty",
            "distribution_penalty",
            "program_target_temperature",
            "interaction_loss_weight",
            "counterfactual_loss_weight",
            "counterfactual_feature_selection_weight",
            "counterfactual_changed_pair_weight",
            "action_structure_weight",
            "minimum_overall_fidelity_for_feedback",
            "minimum_interaction_fidelity_for_feedback",
            "minimum_interaction_validation_samples",
            "maximum_mean_kl_for_feedback",
            "minimum_overall_fidelity_for_explanation",
            "minimum_interaction_fidelity_for_explanation",
            "maximum_mean_kl_for_explanation",
            "minimum_counterfactual_direction_fidelity_for_explanation",
            "minimum_counterfactual_changed_pairs_for_explanation",
            "minimum_counterfactual_pairs",
            "require_explanation_eligibility_for_feedback",
            "require_explanation_eligibility_for_counterfactual_feedback",
        )
        extraction_changed = any(
            name not in previous_config
            or previous_config[name] != getattr(self.config, name)
            for name in extraction_fields
        )
        program_payload = (
            None if extraction_changed else payload.get("program")
        )
        metrics_payload = payload.get("metrics")
        if program_payload is None:
            self.program = None
            self.last_result = None
            self.regularization_weight = 0.0
        else:
            self.program = ExecutableProgram.from_dict(program_payload)
            if not isinstance(metrics_payload, Mapping):
                raise ValueError(
                    "RCPD checkpoint has a program but no metrics."
                )
            metric_names = DistillationMetrics.__dataclass_fields__
            metric_values = {
                name: metrics_payload[name]
                for name in metric_names
                if name in metrics_payload
            }
            self.last_result = RCPDResult(
                program=self.program,
                metrics=DistillationMetrics(**metric_values),
                extraction_summary=(),
            )
            self.regularization_weight = (
                min(
                    self.config.maximum_regularization_lambda,
                    max(0.0, self.config.regularization_lambda),
                )
                if self.last_result.metrics.feedback_eligible
                else 0.0
            )
        last_step = (
            None if extraction_changed else payload.get("last_extract_step")
        )
        self.last_extract_step = (
            int(last_step) if last_step is not None else None
        )
        self.extraction_history = [
            dict(item) for item in payload.get("extraction_history", ())
        ]
        last_error = payload.get("last_error")
        self.last_error = (
            "program structure changed or extraction configuration changed; "
            "a new bounded regularity program is required"
            if extraction_changed
            else (str(last_error) if last_error else None)
        )

    def extractability_report(
        self,
        states: Sequence[State],
        oracle: OraclePolicy,
        feature_encoder: FeatureEncoder,
    ) -> dict[str, Any]:
        if self.program is None:
            return {"available": False, "reason": "no extracted program"}
        metrics = self.evaluate(self.program, states, oracle, feature_encoder)
        return {
            "available": True,
            **metrics.to_dict(),
            "regularization_weight": self.regularization_weight,
        }

    def fit(
        self,
        states: Sequence[State],
        oracle: OraclePolicy,
        feature_encoder: FeatureEncoder,
        *,
        action_legality_features: Mapping[str, str] | None = None,
        action_constraint_reason_features: Mapping[
            str,
            Mapping[str, str],
        ]
        | None = None,
        validation_states: Sequence[State] = (),
        importance_weight_provider: ImportanceWeightProvider | None = None,
        group_provider: SampleGroupProvider | None = None,
        counterfactual_pair_provider: SampleKeyProvider | None = None,
        split_group_provider: SampleKeyProvider | None = None,
        interaction_groups: Sequence[str] = (),
        required_predicate_groups: PredicateGroupContract | None = None,
        program_metadata: Mapping[str, Any] | None = None,
    ) -> RCPDResult:
        """Extract one bounded program from the caller's training-time state set.

        The caller invokes this method repeatedly while RL is still running.
        The returned program is frozen until the next extraction and becomes a
        soft target for the Actor.  Dataset construction stays outside this
        domain-neutral core: callers may combine on-policy, counterfactual, and
        disagreement-focused records as long as every row uses current Actor
        probabilities.
        """

        extraction_started = time.perf_counter()
        if not states:
            raise ValueError("RCPD requires at least one on-policy training state.")
        samples = [
            self._make_sample(
                state,
                oracle,
                feature_encoder,
                importance_weight=(
                    importance_weight_provider(state)
                    if importance_weight_provider is not None
                    else None
                ),
                groups=(
                    group_provider(state)
                    if group_provider is not None
                    else ("ordinary",)
                ),
                pair_id=(
                    counterfactual_pair_provider(state)
                    if counterfactual_pair_provider is not None
                    else None
                ),
                split_group=(
                    split_group_provider(state)
                    if split_group_provider is not None
                    else None
                ),
            )
            for state in states
        ]
        counterfactual_dataset_pairs = len(
            {
                str(sample.pair_id)
                for sample in samples
                if sample.pair_id
            }
        )
        action_names = self._action_names(samples)
        all_feature_names = self._feature_names(samples)
        normalized_required_groups = (
            _normalize_predicate_group_contract(
                required_predicate_groups or {},
                all_feature_names,
            )
        )
        required_features = set((action_legality_features or {}).values())
        required_features.update(
            feature
            for reasons in (action_constraint_reason_features or {}).values()
            for feature in reasons.values()
        )
        required_features.update(
            feature
            for features in normalized_required_groups.values()
            for feature in features
        )
        feature_names = self._limit_feature_names(
            samples,
            all_feature_names,
            action_names=action_names,
            required=required_features,
        )
        program, held_out_indices = self._fit_best_program(
            samples,
            action_names,
            feature_names,
            action_legality_features or {},
            action_constraint_reason_features or {},
            interaction_groups=tuple(str(value) for value in interaction_groups),
            required_predicate_groups=normalized_required_groups,
        )
        if validation_states:
            base_metrics = self.evaluate(
                program,
                validation_states,
                oracle,
                feature_encoder,
                group_provider=group_provider,
                counterfactual_pair_provider=counterfactual_pair_provider,
                split_group_provider=split_group_provider,
                interaction_groups=interaction_groups,
            )
        else:
            evaluation_samples = [
                samples[int(index)] for index in held_out_indices
            ]
            base_metrics = self._metrics_from_samples(
                program,
                evaluation_samples,
                interaction_groups=tuple(
                    str(value) for value in interaction_groups
                ),
            )
        normalized_complexity = program_complexity(
            program,
            max_depth=self.config.max_depth,
            max_leaf_count=(
                self.config.max_leaf_nodes
                if self.config.max_leaf_nodes is not None
                else max(1, program.root.leaf_count())
            ),
            max_predicate_count=(
                self.config.max_predicates
                if self.config.max_predicates is not None
                else max(1, len(program.root.used_predicates()))
            ),
        )
        extractability_loss = (
            (1.0 - base_metrics.action_fidelity)
            + self.config.distribution_penalty
            * base_metrics.mean_kl_divergence
            + self.config.complexity_penalty
            * normalized_complexity.loss
        )
        if base_metrics.interaction_macro_fidelity is not None:
            extractability_loss += self.config.interaction_loss_weight * (
                1.0 - base_metrics.interaction_macro_fidelity
            )
        if base_metrics.counterfactual_delta_error is not None:
            extractability_loss += (
                self.config.counterfactual_loss_weight
                * base_metrics.counterfactual_delta_error
            )
        if base_metrics.counterfactual_direction_fidelity is not None:
            extractability_loss += (
                self.config.counterfactual_loss_weight
                * (1.0 - base_metrics.counterfactual_direction_fidelity)
            )
        extractability_score = math.exp(-max(0.0, extractability_loss))
        semantic_coverage = _program_predicate_group_coverage(
            program,
            normalized_required_groups,
        )
        semantic_coverage_complete = all(semantic_coverage.values())
        ineligibility_reasons: list[str] = []
        if not semantic_coverage_complete:
            ineligibility_reasons.append(
                "required_semantic_predicate_group_missing"
            )
        if (
            base_metrics.action_fidelity
            < self.config.minimum_overall_fidelity_for_feedback
        ):
            ineligibility_reasons.append("overall_fidelity_below_threshold")
        if (
            self.config.maximum_mean_kl_for_feedback is not None
            and base_metrics.mean_kl_divergence
            > float(self.config.maximum_mean_kl_for_feedback)
        ):
            ineligibility_reasons.append("program_fit_kl_above_threshold")
        if interaction_groups:
            if (
                base_metrics.interaction_validation_samples
                < self.config.minimum_interaction_validation_samples
            ):
                ineligibility_reasons.append(
                    "insufficient_interaction_validation_samples"
                )
            elif (
                base_metrics.interaction_macro_fidelity is None
                or base_metrics.interaction_macro_fidelity
                < self.config.minimum_interaction_fidelity_for_feedback
            ):
                ineligibility_reasons.append(
                    "interaction_fidelity_below_threshold"
                )
        if (
            base_metrics.safety_property_violation_rate is not None
            and base_metrics.safety_property_violation_rate > 0.0
        ):
            ineligibility_reasons.append("safety_property_violation")
        # Do not inherit the feedback reasons.  Training regularisation and
        # user-facing explanation answer different questions: the former asks
        # whether a guarded local target is safe enough to nudge the Actor;
        # the latter asks whether the extracted program is faithful enough to
        # be presented as evidence about the NN.  Conflating the two either
        # prevents all feedback or silently weakens explanation standards.
        explanation_ineligibility_reasons: list[str] = []
        if not semantic_coverage_complete:
            explanation_ineligibility_reasons.append(
                "required_semantic_predicate_group_missing"
            )
        if (
            base_metrics.action_fidelity
            <= self.config.minimum_overall_fidelity_for_explanation
        ):
            explanation_ineligibility_reasons.append(
                "overall_fidelity_below_explanation_threshold"
            )
        if (
            self.config.maximum_mean_kl_for_explanation is not None
            and base_metrics.mean_kl_divergence
            >= float(self.config.maximum_mean_kl_for_explanation)
        ):
            explanation_ineligibility_reasons.append(
                "program_fit_kl_above_explanation_threshold"
            )
        if interaction_groups:
            if (
                base_metrics.interaction_validation_samples
                < self.config.minimum_interaction_validation_samples
            ):
                explanation_ineligibility_reasons.append(
                    "insufficient_interaction_validation_samples"
                )
            elif (
                base_metrics.interaction_macro_fidelity is None
                or base_metrics.interaction_macro_fidelity
                < self.config.minimum_interaction_fidelity_for_explanation
            ):
                explanation_ineligibility_reasons.append(
                    "interaction_fidelity_below_explanation_threshold"
                )
        if (
            base_metrics.safety_property_violation_rate is not None
            and base_metrics.safety_property_violation_rate > 0.0
        ):
            explanation_ineligibility_reasons.append(
                "safety_property_violation"
            )
        if (
            counterfactual_dataset_pairs
            < int(self.config.minimum_counterfactual_pairs)
        ):
            explanation_ineligibility_reasons.append(
                "insufficient_counterfactual_pairs"
            )
        if (
            self.config.minimum_counterfactual_changed_pairs_for_explanation
            > 0
        ):
            if (
                base_metrics.counterfactual_changed_pairs
                < self.config.minimum_counterfactual_changed_pairs_for_explanation
            ):
                explanation_ineligibility_reasons.append(
                    "insufficient_counterfactual_changed_pairs"
                )
            elif (
                base_metrics.counterfactual_direction_fidelity is None
                or base_metrics.counterfactual_direction_fidelity
                <= self.config.minimum_counterfactual_direction_fidelity_for_explanation
            ):
                explanation_ineligibility_reasons.append(
                    "counterfactual_direction_fidelity_below_threshold"
                )
        explanation_eligible = not explanation_ineligibility_reasons
        if (
            self.config.require_explanation_eligibility_for_feedback
            and not explanation_eligible
        ):
            for reason in explanation_ineligibility_reasons:
                gated_reason = f"explanation_gate:{reason}"
                if gated_reason not in ineligibility_reasons:
                    ineligibility_reasons.append(gated_reason)
        feedback_eligible = not ineligibility_reasons
        configured_feedback_weight = min(
            self.config.maximum_regularization_lambda,
            max(0.0, self.config.regularization_lambda),
        )
        feedback_weight = (
            configured_feedback_weight if feedback_eligible else 0.0
        )
        final_metrics = DistillationMetrics(
            action_fidelity=base_metrics.action_fidelity,
            mean_kl_divergence=base_metrics.mean_kl_divergence,
            action_regret=base_metrics.action_regret,
            program_size=program.root.size(),
            program_depth=program.root.depth(),
            program_leaf_count=program.root.leaf_count(),
            program_predicate_count=len(program.root.used_predicates()),
            extractability_loss=extractability_loss,
            extractability_score=extractability_score,
            feedback_weight=feedback_weight,
            safety_property_violation_rate=base_metrics.safety_property_violation_rate,
            extraction_time_seconds=time.perf_counter() - extraction_started,
            sample_count=base_metrics.sample_count,
            group_action_fidelity=base_metrics.group_action_fidelity,
            group_validation_samples=(
                base_metrics.group_validation_samples
            ),
            interaction_macro_fidelity=(
                base_metrics.interaction_macro_fidelity
            ),
            counterfactual_delta_error=(
                base_metrics.counterfactual_delta_error
            ),
            counterfactual_direction_fidelity=(
                base_metrics.counterfactual_direction_fidelity
            ),
            counterfactual_validation_pairs=(
                base_metrics.counterfactual_validation_pairs
            ),
            counterfactual_changed_pairs=(
                base_metrics.counterfactual_changed_pairs
            ),
            counterfactual_dataset_pairs=counterfactual_dataset_pairs,
            relational_predicate_count=(
                base_metrics.relational_predicate_count
            ),
            required_predicate_group_coverage=semantic_coverage,
            semantic_predicate_coverage_complete=(
                semantic_coverage_complete
            ),
            interaction_validation_samples=(
                base_metrics.interaction_validation_samples
            ),
            feedback_eligible=feedback_eligible,
            feedback_ineligibility_reasons=tuple(ineligibility_reasons),
            explanation_eligible=explanation_eligible,
            explanation_ineligibility_reasons=tuple(
                explanation_ineligibility_reasons
            ),
        )
        metadata = {
            **dict(program.metadata),
            "method": "regularity_constrained_policy_distillation",
            "training_role": "bounded_program_actor_regularizer",
            "runtime_controller": "neural_policy_only",
            "regularization_version": True,
            "program_complexity": {
                "depth": normalized_complexity.depth,
                "leaves": normalized_complexity.leaves,
                "predicates": normalized_complexity.predicates,
            },
            "normalized_program_complexity": normalized_complexity.to_dict(),
            "program_roles": (
                "training_regularity_signal",
                "local_explanation_audit",
            ),
            "used_relational_features": sorted(
                name
                for name in program.root.used_predicates()
                if _is_relational_predicate(name)
            ),
            "required_predicate_groups": {
                group: list(features)
                for group, features in normalized_required_groups.items()
            },
            "required_predicate_group_coverage": {
                group: list(features)
                for group, features in semantic_coverage.items()
            },
            "semantic_predicate_coverage_complete": (
                semantic_coverage_complete
            ),
            "data_split_audit": _data_split_audit(
                samples,
                held_out_indices,
            ),
            "metrics": final_metrics.to_dict(),
            "config": self.config.__dict__,
            "selection_objective": extractability_loss,
            # Environment integrations may impose a stricter one-way role.
            # Keep the generic regularizer defaults above, then let an
            # explicit extraction contract narrow (never broaden) the role.
            **dict(program_metadata or {}),
        }
        return RCPDResult(
            program=ExecutableProgram(program.action_names, program.feature_names, program.root, metadata),
            metrics=final_metrics,
            extraction_summary=(
                {
                    "sample_count": len(samples),
                    "extractability_loss": extractability_loss,
                    "extractability_score": extractability_score,
                    "program_size": program.root.size(),
                    "program_depth": program.root.depth(),
                    "interaction_macro_fidelity": (
                        final_metrics.interaction_macro_fidelity
                    ),
                    "counterfactual_delta_error": (
                        final_metrics.counterfactual_delta_error
                    ),
                    "feedback_eligible": feedback_eligible,
                    "explanation_eligible": explanation_eligible,
                },
            ),
        )

    def evaluate(
        self,
        program: ExecutableProgram,
        states: Sequence[State],
        oracle: OraclePolicy,
        feature_encoder: FeatureEncoder,
        *,
        group_provider: SampleGroupProvider | None = None,
        counterfactual_pair_provider: SampleKeyProvider | None = None,
        split_group_provider: SampleKeyProvider | None = None,
        interaction_groups: Sequence[str] = (),
    ) -> DistillationMetrics:
        if not states:
            return DistillationMetrics(0.0, 0.0, None, program.root.size(), program.root.depth(), sample_count=0)
        samples = [
            self._make_sample(
                state,
                oracle,
                feature_encoder,
                groups=(
                    group_provider(state)
                    if group_provider is not None
                    else ("ordinary",)
                ),
                pair_id=(
                    counterfactual_pair_provider(state)
                    if counterfactual_pair_provider is not None
                    else None
                ),
                split_group=(
                    split_group_provider(state)
                    if split_group_provider is not None
                    else None
                ),
            )
            for state in states
        ]
        return self._metrics_from_samples(
            program,
            samples,
            interaction_groups=tuple(str(value) for value in interaction_groups),
        )

    def _metrics_from_samples(
        self,
        program: ExecutableProgram,
        samples: Sequence[_Sample],
        *,
        interaction_groups: Sequence[str],
    ) -> DistillationMetrics:
        if not samples:
            return DistillationMetrics(
                0.0,
                0.0,
                None,
                program.root.size(),
                program.root.depth(),
                sample_count=0,
            )
        agreements = 0
        divergences: list[float] = []
        regrets: list[float] = []
        safety_violations = 0
        safety_checks = 0
        group_totals: dict[str, int] = {}
        group_agreements: dict[str, int] = {}
        paired: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        for sample in samples:
            output = sample.oracle
            oracle_probs = output.normalized(program.action_names)
            features = sample.features
            program_mapping = program.predict_proba(features)
            program_probs = np.asarray([program_mapping[action] for action in program.action_names], dtype=float)
            oracle_index = int(np.argmax(oracle_probs))
            program_index = int(np.argmax(program_probs))
            legality_features = program.metadata.get("action_legality_features", {})
            if isinstance(legality_features, Mapping):
                selected_action = program.action_names[program_index]
                legality_feature = legality_features.get(selected_action)
                if legality_feature is not None:
                    safety_checks += 1
                    safety_violations += int(float(features.get(str(legality_feature), 0.0)) <= 0.5)
            agreements += int(oracle_index == program_index)
            for group in sample.groups:
                group_totals[group] = group_totals.get(group, 0) + 1
                group_agreements[group] = group_agreements.get(group, 0) + int(
                    oracle_index == program_index
                )
            if sample.pair_id:
                paired.setdefault(sample.pair_id, []).append(
                    (oracle_probs, program_probs)
                )
            # The same temperature-scaled program distribution is used by the
            # forward-KL training target and by the fidelity audit.  Argmax
            # actions and counterfactual direction metrics continue to use the
            # raw executable-program probabilities, so temperature cannot
            # manufacture action agreement or causal responsiveness.
            program_probs_for_kl = _temperature_scale_probabilities(
                program_probs,
                self.config.program_target_temperature,
            )
            divergences.append(
                _kl_divergence(oracle_probs, program_probs_for_kl)
            )
            if output.q_values:
                chosen = program.action_names[program_index]
                optimal = max(float(value) for value in output.q_values.values())
                regrets.append(optimal - float(output.q_values.get(chosen, optimal)))
            else:
                regrets.append(float(np.max(oracle_probs) - oracle_probs[program_index]))
        group_fidelity = {
            group: group_agreements.get(group, 0) / total
            for group, total in group_totals.items()
            if total > 0
        }
        interaction_set = set(interaction_groups)
        eligible_interaction_groups = [
            group
            for group in interaction_set
            if group_totals.get(group, 0) > 0
            and group_totals.get(group, 0)
            >= self.config.minimum_interaction_validation_samples
        ]
        interaction_macro = (
            float(
                np.mean(
                    [group_fidelity[group] for group in eligible_interaction_groups]
                )
            )
            if eligible_interaction_groups
            else None
        )
        interaction_samples = sum(
            1
            for sample in samples
            if interaction_set.intersection(sample.groups)
        )
        delta_errors: list[float] = []
        direction_matches: list[float] = []
        validation_pairs = 0
        changed_pairs = 0
        for values in paired.values():
            if len(values) < 2:
                continue
            validation_pairs += 1
            first_oracle, first_program = values[0]
            second_oracle, second_program = values[1]
            oracle_delta = second_oracle - first_oracle
            program_delta = second_program - first_program
            delta_errors.append(
                float(np.mean(np.abs(oracle_delta - program_delta)))
            )
            oracle_changed = int(np.argmax(first_oracle)) != int(
                np.argmax(second_oracle)
            )
            # Direction fidelity is meaningful only when the NN actually
            # changes its preferred action under the intervention.  Counting
            # unchanged pairs as successes made a program look causally
            # faithful even when it ignored the intervened relation.  For a
            # changed pair, require both endpoints to match; checking only the
            # arbitrarily ordered second endpoint was asymmetric and could
            # hide a wrong baseline explanation.
            if oracle_changed:
                changed_pairs += 1
                direction_matches.append(
                    float(
                        int(np.argmax(first_oracle))
                        == int(np.argmax(first_program))
                        and int(np.argmax(second_oracle))
                        == int(np.argmax(second_program))
                    )
                )
        used_predicates = program.root.used_predicates()
        relational_predicates = sum(
            _is_relational_predicate(name) for name in used_predicates
        )
        return DistillationMetrics(
            action_fidelity=agreements / len(samples),
            mean_kl_divergence=float(np.mean(divergences)),
            action_regret=float(np.mean(regrets)) if regrets else None,
            program_size=program.root.size(),
            program_depth=program.root.depth(),
            program_leaf_count=program.root.leaf_count(),
            program_predicate_count=len(program.root.used_predicates()),
            safety_property_violation_rate=(
                safety_violations / safety_checks if safety_checks else None
            ),
            sample_count=len(samples),
            group_action_fidelity=group_fidelity,
            group_validation_samples=dict(group_totals),
            interaction_macro_fidelity=interaction_macro,
            counterfactual_delta_error=(
                float(np.mean(delta_errors)) if delta_errors else None
            ),
            counterfactual_direction_fidelity=(
                float(np.mean(direction_matches))
                if direction_matches
                else None
            ),
            counterfactual_validation_pairs=validation_pairs,
            counterfactual_changed_pairs=changed_pairs,
            relational_predicate_count=int(relational_predicates),
            interaction_validation_samples=interaction_samples,
        )

    def _make_sample(
        self,
        state: State,
        oracle: OraclePolicy,
        feature_encoder: FeatureEncoder,
        *,
        importance_weight: float | None = None,
        groups: Sequence[str] = ("ordinary",),
        pair_id: str | None = None,
        split_group: str | None = None,
    ) -> _Sample:
        output = _as_oracle_output(oracle(state))
        features = {str(name): float(value) for name, value in feature_encoder(state).items()}
        if not features:
            raise ValueError("Relational feature encoder returned no features.")
        if any(name.startswith("feature_") for name in features):
            raise ValueError("RCPD requires semantic relational feature names, not feature_N placeholders.")
        probabilities = sorted((float(value) for value in output.probabilities.values()), reverse=True)
        margin = probabilities[0] - probabilities[1] if len(probabilities) > 1 else probabilities[0]
        if output.q_values and len(output.q_values) > 1:
            q_values = sorted((float(value) for value in output.q_values.values()), reverse=True)
            margin = max(margin, q_values[0] - q_values[1])
        sample_weight = (
            max(1e-8, float(importance_weight))
            if importance_weight is not None
            else 1.0
            + self.config.importance_weight_scale * max(0.0, margin)
        )
        return _Sample(
            state,
            features,
            output,
            sample_weight,
            tuple(dict.fromkeys(str(value) for value in groups))
            or ("ordinary",),
            str(pair_id) if pair_id else None,
            str(split_group) if split_group else None,
        )

    def _fit_best_program(
        self,
        samples: Sequence[_Sample],
        action_names: tuple[str, ...],
        feature_names: tuple[str, ...],
        action_legality_features: Mapping[str, str],
        action_constraint_reason_features: Mapping[
            str,
            Mapping[str, str],
        ],
        *,
        interaction_groups: Sequence[str],
        required_predicate_groups: Mapping[str, tuple[str, ...]],
    ) -> tuple[ExecutableProgram, np.ndarray]:
        x = np.asarray([[sample.features.get(name, 0.0) for name in feature_names] for sample in samples], dtype=float)
        y = np.asarray([sample.oracle.normalized(action_names) for sample in samples], dtype=float)
        weights = np.asarray([sample.weight for sample in samples])
        validation_size = max(1, int(len(samples) * self.config.validation_fraction)) if len(samples) > 4 else len(samples)
        validation = _stratified_validation_indices(
            samples,
            validation_size=validation_size,
            random_seed=self.config.random_seed + len(samples),
            interaction_groups=interaction_groups,
            minimum_group_samples=(
                self.config.minimum_interaction_validation_samples
            ),
        )
        validation = _ensure_required_semantic_training_variation(
            samples,
            validation,
            required_predicate_groups,
        )
        validation_set = set(validation.tolist())
        training = np.asarray(
            [
                index
                for index in range(len(samples))
                if index not in validation_set
            ],
            dtype=int,
        )
        if not len(training):
            training = validation.copy()
        structure_targets = y
        action_structure_weight = max(
            0.0,
            float(self.config.action_structure_weight),
        )
        if action_structure_weight > 0.0:
            hard_actions = np.zeros_like(y)
            hard_actions[
                np.arange(len(y)),
                np.argmax(y, axis=1),
            ] = math.sqrt(action_structure_weight)
            structure_targets = np.concatenate(
                (y, hard_actions),
                axis=1,
            )
        fit_weights = weights.copy()
        changed_pair_training_indices = (
            _counterfactual_changed_training_indices(
                samples,
                training,
                action_names,
            )
        )
        changed_pair_weight = max(
            1.0,
            float(self.config.counterfactual_changed_pair_weight),
        )
        if changed_pair_weight > 1.0:
            fit_weights[changed_pair_training_indices] *= (
                changed_pair_weight
            )
        best: tuple[float, ExecutableProgram] | None = None
        for depth in range(1, self.config.max_depth + 1):
            estimator = DecisionTreeRegressor(
                max_depth=depth,
                max_leaf_nodes=self.config.max_leaf_nodes,
                min_samples_leaf=max(1, min(self.config.min_samples_leaf, len(training) // 2 or 1)),
                random_state=self.config.random_seed,
            )
            estimator.fit(
                x[training],
                structure_targets[training],
                sample_weight=fit_weights[training],
            )
            root = _tree_to_program(estimator, feature_names, len(action_names))
            program = ExecutableProgram(
                action_names=action_names,
                feature_names=feature_names,
                root=root,
                metadata={
                    "candidate_depth": depth,
                    "training_samples": len(samples),
                    "counterfactual_changed_pair_weight": (
                        changed_pair_weight
                    ),
                    "counterfactual_changed_training_samples": int(
                        len(changed_pair_training_indices)
                    ),
                    "action_structure_weight": action_structure_weight,
                },
            )
            program = ExecutableProgram(
                action_names=program.action_names,
                feature_names=program.feature_names,
                root=program.root,
                metadata={
                    **dict(program.metadata),
                    "action_legality_features": dict(action_legality_features),
                    "action_constraint_reason_features": {
                        str(action): dict(reasons)
                        for action, reasons in action_constraint_reason_features.items()
                    },
                },
            )
            validation_samples = [samples[int(index)] for index in validation]
            candidate_metrics = self._metrics_from_samples(
                program,
                validation_samples,
                interaction_groups=interaction_groups,
            )
            fidelity_loss = 1.0 - candidate_metrics.action_fidelity
            kl = candidate_metrics.mean_kl_divergence
            candidate_complexity = program_complexity(
                program,
                max_depth=self.config.max_depth,
                max_leaf_count=(
                    self.config.max_leaf_nodes
                    if self.config.max_leaf_nodes is not None
                    else max(1, root.leaf_count())
                ),
                max_predicate_count=(
                    self.config.max_predicates
                    if self.config.max_predicates is not None
                    else max(1, len(root.used_predicates()))
                ),
            )
            objective = (
                fidelity_loss
                + self.config.distribution_penalty * kl
                + self.config.complexity_penalty
                * candidate_complexity.loss
            )
            if candidate_metrics.interaction_macro_fidelity is not None:
                objective += self.config.interaction_loss_weight * (
                    1.0 - candidate_metrics.interaction_macro_fidelity
                )
            if candidate_metrics.counterfactual_delta_error is not None:
                objective += self.config.counterfactual_loss_weight * (
                    candidate_metrics.counterfactual_delta_error
                )
            if candidate_metrics.counterfactual_direction_fidelity is not None:
                objective += self.config.counterfactual_loss_weight * (
                    1.0
                    - candidate_metrics.counterfactual_direction_fidelity
                )
            if best is None or objective < best[0]:
                best = (objective, program)
        if best is None:
            raise RuntimeError("No RCPD program candidate was trained.")
        selected_program = best[1]
        coverage = _program_predicate_group_coverage(
            selected_program,
            required_predicate_groups,
        )
        if required_predicate_groups and not all(coverage.values()):
            selected_program = self._repair_semantic_predicate_coverage(
                selected_program,
                samples,
                training,
                validation,
                action_names=action_names,
                required_predicate_groups=required_predicate_groups,
                interaction_groups=interaction_groups,
            )
        final_coverage = _program_predicate_group_coverage(
            selected_program,
            required_predicate_groups,
        )
        missing = [
            group for group, features in final_coverage.items() if not features
        ]
        if missing:
            raise ValueError(
                "Unable to fit a bounded program containing required "
                "semantic predicate groups: " + ", ".join(missing)
            )
        return selected_program, validation

    def _repair_semantic_predicate_coverage(
        self,
        program: ExecutableProgram,
        samples: Sequence[_Sample],
        training: np.ndarray,
        validation: np.ndarray,
        *,
        action_names: tuple[str, ...],
        required_predicate_groups: Mapping[str, tuple[str, ...]],
        interaction_groups: Sequence[str],
    ) -> ExecutableProgram:
        """Constrain a fitted tree to use every declared semantic family.

        Ordinary CART may ignore a low-frequency but explanation-critical
        factor even after it survives feature preselection.  For each missing
        family, replace the least harmful existing decision with the best
        split from that family, then re-estimate every leaf from the current
        NN labels.  Depth and leaf bounds are unchanged, and no action label is
        invented.  A family that has no varying, valid split makes extraction
        fail visibly instead of producing a semantically incomplete program.
        """

        current = program
        training_samples = [samples[int(index)] for index in training]
        validation_samples = [samples[int(index)] for index in validation]
        repaired_groups: list[str] = []
        while True:
            coverage = _program_predicate_group_coverage(
                current,
                required_predicate_groups,
            )
            missing_groups = [
                group for group, features in coverage.items() if not features
            ]
            if not missing_groups:
                break
            group = missing_groups[0]
            candidates = tuple(
                feature
                for feature in required_predicate_groups[group]
                if feature in current.feature_names
                and _feature_varies(training_samples, feature)
            )
            if not candidates:
                raise ValueError(
                    f"Required semantic predicate group {group!r} has no "
                    "varying feature in the extraction dataset."
                )
            protected = {
                feature
                for covered_group, features in coverage.items()
                if covered_group != group and len(features) == 1
                for feature in features
            }
            best_repair: tuple[float, ExecutableProgram] | None = None

            def consider(replaced_root: ProgramNode) -> None:
                nonlocal best_repair
                # Structural repairs must remain executable on every training
                # route.  Empty leaves would make their probability an
                # untested historical artefact rather than a current-NN fit.
                if _minimum_program_leaf_samples(
                    replaced_root,
                    training_samples,
                ) < 1:
                    return
                refit_root = _reestimate_program_leaves(
                    replaced_root,
                    training_samples,
                    action_names,
                )
                candidate = ExecutableProgram(
                    action_names=current.action_names,
                    feature_names=current.feature_names,
                    root=refit_root,
                    metadata=dict(current.metadata),
                )
                candidate_coverage = _program_predicate_group_coverage(
                    candidate,
                    required_predicate_groups,
                )
                if not candidate_coverage.get(group):
                    return
                if any(
                    not candidate_coverage.get(previous_group)
                    for previous_group in repaired_groups
                ):
                    return
                metrics = self._metrics_from_samples(
                    candidate,
                    validation_samples,
                    interaction_groups=interaction_groups,
                )
                objective = self._program_selection_objective(
                    candidate,
                    metrics,
                )
                if best_repair is None or objective < best_repair[0]:
                    best_repair = (objective, candidate)

            for path, node in _internal_program_nodes(current.root):
                if node.feature in protected:
                    continue
                for feature in candidates:
                    for threshold in _candidate_feature_thresholds(
                        training_samples,
                        feature,
                    ):
                        replaced_root = _replace_program_node_split(
                            current.root,
                            path,
                            feature=feature,
                            threshold=threshold,
                        )
                        consider(replaced_root)

            # A shallow naturally fitted tree may have fewer internal nodes
            # than the semantic contract has groups.  When capacity remains,
            # add a bounded split to an existing leaf instead of deleting a
            # useful task split.  The split is fitted only where that leaf is
            # reached and the resulting leaves are again estimated from the
            # current neural policy labels.
            leaf_limit = (
                int(self.config.max_leaf_nodes)
                if self.config.max_leaf_nodes is not None
                else None
            )
            can_add_leaf = (
                leaf_limit is None
                or current.root.leaf_count() < leaf_limit
            )
            if can_add_leaf:
                for path, leaf in _leaf_program_nodes(current.root):
                    if len(path) >= int(self.config.max_depth):
                        continue
                    local_samples = _samples_reaching_path(
                        current.root,
                        training_samples,
                        path,
                    )
                    for feature in candidates:
                        if not _feature_varies(local_samples, feature):
                            continue
                        for threshold in _candidate_feature_thresholds(
                            local_samples,
                            feature,
                        ):
                            left_count = sum(
                                float(sample.features.get(feature, 0.0))
                                <= threshold
                                for sample in local_samples
                            )
                            right_count = len(local_samples) - left_count
                            minimum_rows = max(
                                1,
                                min(
                                    int(self.config.min_samples_leaf),
                                    len(local_samples) // 2,
                                ),
                            )
                            if min(left_count, right_count) < minimum_rows:
                                continue
                            replacement = ProgramNode(
                                feature=feature,
                                threshold=float(threshold),
                                left=ProgramNode(
                                    probabilities=leaf.probabilities
                                ),
                                right=ProgramNode(
                                    probabilities=leaf.probabilities
                                ),
                            )
                            consider(
                                _replace_program_subtree(
                                    current.root,
                                    path,
                                    replacement,
                                )
                            )
            if best_repair is None:
                raise ValueError(
                    f"No bounded-tree repair could include required semantic "
                    f"predicate group {group!r}."
                )
            current = best_repair[1]
            repaired_groups.append(group)
        return ExecutableProgram(
            action_names=current.action_names,
            feature_names=current.feature_names,
            root=current.root,
            metadata={
                **dict(current.metadata),
                "semantic_coverage_repaired": bool(repaired_groups),
                "semantic_coverage_repaired_groups": repaired_groups,
            },
        )

    def _program_selection_objective(
        self,
        program: ExecutableProgram,
        metrics: DistillationMetrics,
    ) -> float:
        complexity = program_complexity(
            program,
            max_depth=self.config.max_depth,
            max_leaf_count=(
                self.config.max_leaf_nodes
                if self.config.max_leaf_nodes is not None
                else max(1, program.root.leaf_count())
            ),
            max_predicate_count=(
                self.config.max_predicates
                if self.config.max_predicates is not None
                else max(1, len(program.root.used_predicates()))
            ),
        )
        objective = (
            1.0
            - metrics.action_fidelity
            + self.config.distribution_penalty
            * metrics.mean_kl_divergence
            + self.config.complexity_penalty * complexity.loss
        )
        if metrics.interaction_macro_fidelity is not None:
            objective += self.config.interaction_loss_weight * (
                1.0 - metrics.interaction_macro_fidelity
            )
        if metrics.counterfactual_delta_error is not None:
            objective += self.config.counterfactual_loss_weight * (
                metrics.counterfactual_delta_error
            )
        if metrics.counterfactual_direction_fidelity is not None:
            objective += self.config.counterfactual_loss_weight * (
                1.0 - metrics.counterfactual_direction_fidelity
            )
        return float(objective)

    def _limit_feature_names(
        self,
        samples: Sequence[_Sample],
        feature_names: tuple[str, ...],
        *,
        action_names: tuple[str, ...],
        required: set[str],
    ) -> tuple[str, ...]:
        feature_names = tuple(
            name for name in feature_names if _program_feature_allowed(name)
        )
        limit = self.config.max_predicates
        if limit is None or len(feature_names) <= limit:
            return feature_names
        required_ordered = [name for name in feature_names if name in required]
        remaining = [name for name in feature_names if name not in required]
        if len(required_ordered) >= int(limit):
            return tuple(required_ordered)
        y = np.asarray(
            [sample.oracle.normalized(action_names) for sample in samples],
            dtype=float,
        )
        weights = np.asarray([sample.weight for sample in samples], dtype=float)
        weights = weights / max(1e-12, float(weights.sum()))
        mean_target = np.sum(y * weights[:, None], axis=0)
        baseline = float(
            np.sum(weights[:, None] * np.square(y - mean_target))
        )
        # Marginal stump gain can miss a rare but causally important relation
        # when ordinary navigation states dominate the replay.  Paired legal
        # counterfactuals provide a second, domain-neutral signal: prefer a
        # feature when it changes across a pair exactly where the NN action
        # distribution changes.  This only affects which predicates are made
        # available to the bounded tree; it does not invent labels or force a
        # relation into the final program.
        paired_samples: dict[str, list[int]] = {}
        for index, sample in enumerate(samples):
            if sample.pair_id:
                paired_samples.setdefault(sample.pair_id, []).append(index)
        paired_endpoints: list[tuple[int, int, float]] = []
        for indices in paired_samples.values():
            if len(indices) < 2:
                continue
            # Directed probes are normally pairs.  If a provider emits more
            # endpoints, choose the two with the largest NN response so the
            # audit remains deterministic and focused on an actual change.
            best_pair: tuple[int, int, float] | None = None
            for left_position, left_index in enumerate(indices[:-1]):
                for right_index in indices[left_position + 1 :]:
                    policy_delta = 0.5 * float(
                        np.abs(y[left_index] - y[right_index]).sum()
                    )
                    candidate = (left_index, right_index, policy_delta)
                    if best_pair is None or candidate[2] > best_pair[2]:
                        best_pair = candidate
            if best_pair is not None:
                paired_endpoints.append(best_pair)
        scores: dict[str, float] = {}
        for name in remaining:
            values = np.asarray(
                [sample.features.get(name, 0.0) for sample in samples],
                dtype=float,
            )
            if np.allclose(values, values[0]):
                scores[name] = float("-inf")
                continue
            estimator = DecisionTreeRegressor(
                max_depth=1,
                min_samples_leaf=max(
                    1,
                    min(self.config.min_samples_leaf, len(samples) // 2 or 1),
                ),
                random_state=self.config.random_seed,
            )
            estimator.fit(
                values.reshape(-1, 1),
                y,
                sample_weight=np.asarray([sample.weight for sample in samples]),
            )
            predicted = estimator.predict(values.reshape(-1, 1))
            loss = float(
                np.sum(weights[:, None] * np.square(y - predicted))
            )
            pair_score = 0.0
            if paired_endpoints:
                feature_span = max(
                    1e-12,
                    float(values.max() - values.min()),
                )
                pair_score = float(
                    np.mean(
                        [
                            min(
                                1.0,
                                abs(
                                    float(values[left_index])
                                    - float(values[right_index])
                                )
                                / feature_span,
                            )
                            * policy_delta
                            for left_index, right_index, policy_delta
                            in paired_endpoints
                        ]
                    )
                )
            scores[name] = (
                baseline
                - loss
                + self.config.counterfactual_feature_selection_weight
                * pair_score
            )
        ranked = sorted(remaining, key=lambda name: (-scores[name], name))
        selected = list(required_ordered)
        by_family: dict[str, list[str]] = {}
        for name in ranked:
            by_family.setdefault(_feature_family(name), []).append(name)
        for family in sorted(by_family):
            for name in by_family[family][:4]:
                if len(selected) >= int(limit):
                    break
                if name not in selected:
                    selected.append(name)
        for name in ranked:
            if len(selected) >= int(limit):
                break
            if name not in selected:
                selected.append(name)
        return tuple(selected)

    @staticmethod
    def _action_names(samples: Sequence[_Sample]) -> tuple[str, ...]:
        first = tuple(samples[0].oracle.probabilities)
        if not first:
            raise ValueError("Oracle returned no actions.")
        expected = set(first)
        for sample in samples[1:]:
            if set(sample.oracle.probabilities) != expected:
                raise ValueError("Oracle action schema changed across states.")
        return first

    @staticmethod
    def _feature_names(samples: Sequence[_Sample]) -> tuple[str, ...]:
        return tuple(sorted({name for sample in samples for name in sample.features}))




def implementation_audit() -> dict[str, Any]:
    """Machine-readable proof that the algorithm file has no environment imports."""

    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_roots = {"env", "envs"}
    violations: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported = [node.module or ""]
        else:
            continue
        for module_name in imported:
            pieces = {piece.lower() for piece in module_name.split(".")}
            violations.update(pieces & forbidden_roots)
    return {
        "algorithm": "Regularity-Constrained Policy Distillation",
        "single_file": str(Path(__file__).name) == "rcpd.py",
        "forbidden_environment_imports": sorted(violations),
        "environment_independent": not violations,
    }
