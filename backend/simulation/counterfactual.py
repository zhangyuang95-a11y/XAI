"""Paired counterfactual simulation for the policy used by the explanation agent."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np

from backend.adapters.base import (
    CandidateIntervention,
    EnvironmentAdapter,
    EnvironmentSnapshot,
    Intervention,
    PolicyProtocol,
    RolloutResult,
)


@dataclass(frozen=True)
class PairedRollout:
    seed: int
    baseline: RolloutResult
    counterfactual: RolloutResult


@dataclass(frozen=True)
class CounterfactualResult:
    original_snapshot: EnvironmentSnapshot
    intervened_snapshot: EnvironmentSnapshot
    interventions: tuple[Intervention, ...]
    # Backward-compatible name: these are the counterfactual branch rollouts.
    rollouts: tuple[RolloutResult, ...]
    first_action_distribution: Mapping[str, Mapping[str, float]]
    baseline_rollouts: tuple[RolloutResult, ...] = ()
    baseline_first_action_distribution: Mapping[str, Mapping[str, float]] = field(
        default_factory=dict
    )
    action_probability_delta: Mapping[str, Mapping[str, float]] = field(
        default_factory=dict
    )
    paired_rollouts: tuple[PairedRollout, ...] = ()
    paired_seeds: tuple[int, ...] = ()

    def changed_action_probability(
        self,
        agent_id: str,
        action: str,
        *,
        minimum_effect: float = 0.0,
    ) -> bool:
        return abs(self.action_probability_delta.get(agent_id, {}).get(action, 0.0)) > minimum_effect


@dataclass(frozen=True)
class CandidateEffect:
    """Measured effect of one adapter-proposed factor on the explanation policy."""

    candidate: CandidateIntervention
    observable_by_actor: bool
    l1_policy_effect: float
    action_probability_delta: Mapping[str, float]
    action_change_rate: float
    supported: bool
    counterfactual: CounterfactualResult


@dataclass(frozen=True)
class WhyAnalysis:
    target_entity: str
    baseline_action: str
    candidates: tuple[CandidateEffect, ...]
    supported_candidates: tuple[CandidateEffect, ...]
    minimum_effect: float


@dataclass(frozen=True)
class WhyNotRecourse:
    target_entity: str
    desired_action: str
    baseline_probability: float
    achieved: bool
    selected: CandidateEffect | None
    candidates: tuple[CandidateEffect, ...]
    refusal_reason: str | None = None


class CounterfactualEngine:
    """Run paired branches with the policy supplied by the query engine.

    In the main system this is the extracted Python policy, so open-ended and
    counterfactual answers are derived from the same inspectable program.  A
    neural policy can still be supplied explicitly for baseline experiments.
    """

    def __init__(self, adapter: EnvironmentAdapter, policy: PolicyProtocol) -> None:
        self.adapter = adapter
        self.policy = policy

    def simulate(
        self,
        snapshot: EnvironmentSnapshot,
        interventions: Sequence[Intervention],
        *,
        horizon: int,
        repetitions: int = 1,
        deterministic: bool = False,
        seed: int = 2026,
    ) -> CounterfactualResult:
        intervened = self.adapter.apply_interventions(snapshot, interventions)
        baseline_rollouts: list[RolloutResult] = []
        counterfactual_rollouts: list[RolloutResult] = []
        pairs: list[PairedRollout] = []
        seeds = tuple(seed + repetition for repetition in range(max(1, repetitions)))
        try:
            for pair_seed in seeds:
                baseline = self._branch(
                    snapshot,
                    horizon=horizon,
                    deterministic=deterministic,
                    seed=pair_seed,
                )
                counterfactual = self._branch(
                    intervened,
                    horizon=horizon,
                    deterministic=deterministic,
                    seed=pair_seed,
                )
                baseline_rollouts.append(baseline)
                counterfactual_rollouts.append(counterfactual)
                pairs.append(PairedRollout(pair_seed, baseline, counterfactual))
        finally:
            self.adapter.restore(snapshot, self.policy)

        baseline_distribution = _mean_first_policy_distribution(baseline_rollouts)
        counterfactual_distribution = _mean_first_policy_distribution(
            counterfactual_rollouts
        )
        delta = _distribution_delta(baseline_distribution, counterfactual_distribution)
        return CounterfactualResult(
            original_snapshot=snapshot,
            intervened_snapshot=intervened,
            interventions=tuple(interventions),
            rollouts=tuple(counterfactual_rollouts),
            first_action_distribution=counterfactual_distribution,
            baseline_rollouts=tuple(baseline_rollouts),
            baseline_first_action_distribution=baseline_distribution,
            action_probability_delta=delta,
            paired_rollouts=tuple(pairs),
            paired_seeds=seeds,
        )

    def analyze_why(
        self,
        snapshot: EnvironmentSnapshot,
        target_entity: str,
        *,
        repetitions: int = 4,
        seed: int = 2026,
        minimum_effect: float = 0.02,
        max_candidates: int = 24,
        test_combinations: bool = True,
    ) -> WhyAnalysis:
        """Test candidate causes; no adapter candidate is presumed causal."""

        baseline_outputs = self.adapter.policy_outputs(snapshot, self.policy)
        if target_entity not in baseline_outputs:
            raise ValueError(f"Unknown policy-controlled entity: {target_entity}")
        baseline_action = baseline_outputs[target_entity].argmax_action
        candidates = list(
            self.adapter.causal_intervention_candidates(
                snapshot,
                target_entity,
            )
        )[: max(0, int(max_candidates))]
        measured = [
            self._measure_candidate(
                snapshot,
                target_entity,
                candidate,
                repetitions=repetitions,
                seed=seed + index * 1009,
                minimum_effect=minimum_effect,
            )
            for index, candidate in enumerate(candidates)
        ]
        if test_combinations and not any(item.supported for item in measured):
            ranked = sorted(
                measured,
                key=lambda item: item.l1_policy_effect,
                reverse=True,
            )[:4]
            for pair_index, (left, right) in enumerate(
                combinations(ranked, 2)
            ):
                combined = _combine_candidates(left.candidate, right.candidate)
                if combined is None:
                    continue
                valid, _ = self.adapter.validate_intervention(
                    snapshot, combined.interventions
                )
                if not valid:
                    continue
                measured.append(
                    self._measure_candidate(
                        snapshot,
                        target_entity,
                        combined,
                        repetitions=repetitions,
                        seed=seed + 50000 + pair_index * 1009,
                        minimum_effect=minimum_effect,
                    )
                )
        ordered = tuple(
            sorted(
                measured,
                key=lambda item: (
                    not item.supported,
                    -item.l1_policy_effect,
                    len(item.candidate.interventions),
                    item.candidate.candidate_id,
                ),
            )
        )
        return WhyAnalysis(
            target_entity=target_entity,
            baseline_action=baseline_action,
            candidates=ordered,
            supported_candidates=tuple(
                item for item in ordered if item.supported
            ),
            minimum_effect=float(minimum_effect),
        )

    def search_why_not(
        self,
        snapshot: EnvironmentSnapshot,
        target_entity: str,
        desired_action: str,
        *,
        repetitions: int = 4,
        seed: int = 2026,
        minimum_effect: float = 0.01,
        max_candidates: int = 24,
    ) -> WhyNotRecourse:
        """Search minimal legal edits that make an alternative action dominant."""

        action_names = tuple(self.policy.action_names)
        if desired_action not in action_names:
            return WhyNotRecourse(
                target_entity=target_entity,
                desired_action=desired_action,
                baseline_probability=0.0,
                achieved=False,
                selected=None,
                candidates=(),
                refusal_reason=(
                    f"Desired action {desired_action!r} is outside the policy action schema."
                ),
            )
        baseline_outputs = self.adapter.policy_outputs(snapshot, self.policy)
        if target_entity not in baseline_outputs:
            return WhyNotRecourse(
                target_entity=target_entity,
                desired_action=desired_action,
                baseline_probability=0.0,
                achieved=False,
                selected=None,
                candidates=(),
                refusal_reason=f"Unknown policy-controlled entity: {target_entity}",
            )
        baseline_distribution = baseline_outputs[target_entity]
        baseline_probability = dict(
            zip(
                baseline_distribution.actions,
                baseline_distribution.probabilities,
            )
        ).get(desired_action, 0.0)
        candidates = list(
            self.adapter.recourse_intervention_candidates(
                snapshot,
                target_entity,
                desired_action,
            )
        )[: max(0, int(max_candidates))]
        measured = [
            self._measure_candidate(
                snapshot,
                target_entity,
                candidate,
                repetitions=repetitions,
                seed=seed + index * 1013,
                minimum_effect=minimum_effect,
            )
            for index, candidate in enumerate(candidates)
        ]

        def succeeds(item: CandidateEffect) -> bool:
            distribution = item.counterfactual.first_action_distribution.get(
                target_entity, {}
            )
            if not distribution:
                return False
            return (
                item.observable_by_actor
                and max(distribution, key=distribution.__getitem__)
                == desired_action
                and float(distribution.get(desired_action, 0.0))
                > baseline_probability + minimum_effect
            )

        successful = [item for item in measured if succeeds(item)]
        if not successful:
            ranked = sorted(
                measured,
                key=lambda item: item.action_probability_delta.get(
                    desired_action, 0.0
                ),
                reverse=True,
            )[:4]
            for pair_index, (left, right) in enumerate(
                combinations(ranked, 2)
            ):
                combined = _combine_candidates(left.candidate, right.candidate)
                if combined is None:
                    continue
                valid, _ = self.adapter.validate_intervention(
                    snapshot, combined.interventions
                )
                if not valid:
                    continue
                item = self._measure_candidate(
                    snapshot,
                    target_entity,
                    combined,
                    repetitions=repetitions,
                    seed=seed + 70000 + pair_index * 1013,
                    minimum_effect=minimum_effect,
                )
                measured.append(item)
                if succeeds(item):
                    successful.append(item)
        selected = (
            min(
                successful,
                key=lambda item: (
                    len(item.candidate.interventions),
                    -item.counterfactual.first_action_distribution.get(
                        target_entity, {}
                    ).get(desired_action, 0.0),
                    item.candidate.candidate_id,
                ),
            )
            if successful
            else None
        )
        ordered = tuple(
            sorted(
                measured,
                key=lambda item: (
                    len(item.candidate.interventions),
                    -item.action_probability_delta.get(desired_action, 0.0),
                    item.candidate.candidate_id,
                ),
            )
        )
        return WhyNotRecourse(
            target_entity=target_entity,
            desired_action=desired_action,
            baseline_probability=float(baseline_probability),
            achieved=selected is not None,
            selected=selected,
            candidates=ordered,
            refusal_reason=(
                None
                if selected is not None
                else "No tested legal state change made the requested alternative action dominant."
            ),
        )

    def _measure_candidate(
        self,
        snapshot: EnvironmentSnapshot,
        target_entity: str,
        candidate: CandidateIntervention,
        *,
        repetitions: int,
        seed: int,
        minimum_effect: float,
    ) -> CandidateEffect:
        result = self.simulate(
            snapshot,
            candidate.interventions,
            horizon=1,
            repetitions=max(2, repetitions),
            deterministic=False,
            seed=seed,
        )
        before = np.asarray(snapshot.observations[target_entity])
        after = np.asarray(
            result.intervened_snapshot.observations[target_entity]
        )
        observable = not np.array_equal(before, after)
        delta = dict(
            result.action_probability_delta.get(target_entity, {})
        )
        l1_effect = sum(abs(float(value)) for value in delta.values())
        return CandidateEffect(
            candidate=candidate,
            observable_by_actor=observable,
            l1_policy_effect=float(l1_effect),
            action_probability_delta=delta,
            action_change_rate=_paired_action_change_rate(
                result.paired_rollouts,
                target_entity,
            ),
            supported=bool(observable and l1_effect >= minimum_effect),
            counterfactual=result,
        )

    def _branch(
        self,
        snapshot: EnvironmentSnapshot,
        *,
        horizon: int,
        deterministic: bool,
        seed: int,
    ) -> RolloutResult:
        self.adapter.restore(snapshot, self.policy)
        if not deterministic and hasattr(self.policy, "seed_rng"):
            self.policy.seed_rng(seed)
        return self.adapter.rollout(
            self.policy,
            horizon=horizon,
            deterministic=deterministic,
        )


def _mean_first_policy_distribution(
    rollouts: Sequence[RolloutResult],
) -> dict[str, dict[str, float]]:
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, int] = defaultdict(int)
    for rollout in rollouts:
        if not rollout.frames:
            continue
        for agent_id, distribution in rollout.frames[0].distributions.items():
            counts[agent_id] += 1
            for action, probability in zip(
                distribution.actions,
                distribution.probabilities,
            ):
                totals[agent_id][action] += float(probability)
    return {
        agent_id: {
            action: value / max(1, counts[agent_id])
            for action, value in action_totals.items()
        }
        for agent_id, action_totals in totals.items()
    }


def _distribution_delta(
    baseline: Mapping[str, Mapping[str, float]],
    counterfactual: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for agent_id in sorted(set(baseline) | set(counterfactual)):
        baseline_actions = baseline.get(agent_id, {})
        counterfactual_actions = counterfactual.get(agent_id, {})
        result[agent_id] = {
            action: float(counterfactual_actions.get(action, 0.0))
            - float(baseline_actions.get(action, 0.0))
            for action in sorted(set(baseline_actions) | set(counterfactual_actions))
        }
    return result


def _paired_action_change_rate(
    pairs: Sequence[PairedRollout],
    target_entity: str,
) -> float:
    comparisons = 0
    changed = 0
    for pair in pairs:
        if not pair.baseline.frames or not pair.counterfactual.frames:
            continue
        baseline = pair.baseline.frames[0].proposed_actions.get(target_entity)
        counterfactual = pair.counterfactual.frames[0].proposed_actions.get(
            target_entity
        )
        if baseline is None or counterfactual is None:
            continue
        comparisons += 1
        changed += int(baseline != counterfactual)
    return changed / max(1, comparisons)


def _combine_candidates(
    left: CandidateIntervention,
    right: CandidateIntervention,
) -> CandidateIntervention | None:
    edits = (*left.interventions, *right.interventions)
    targets: dict[tuple[str, str], Any] = {}
    for edit in edits:
        key = (edit.entity_id, edit.property_name)
        if key in targets and targets[key] != edit.value:
            return None
        targets[key] = edit.value
    unique = tuple(
        Intervention(entity_id, property_name, value)
        for (entity_id, property_name), value in targets.items()
    )
    return CandidateIntervention(
        candidate_id=(
            f"combined:{left.candidate_id}+{right.candidate_id}"
        ),
        factor=f"{left.factor}+{right.factor}",
        description=f"{left.description} {right.description}",
        interventions=unique,
        provenance={
            "combined": True,
            "components": (
                left.candidate_id,
                right.candidate_id,
            ),
            "actor_visible": bool(
                left.provenance.get("actor_visible", False)
                and right.provenance.get("actor_visible", False)
            ),
        },
    )
