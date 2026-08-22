"""Domain-neutral contracts for policies, environments, and interventions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from core.policy_contracts import ActionDistribution


@dataclass(frozen=True)
class Intervention:
    """One validated change applied to a restored environment state."""

    entity_id: str
    property_name: str
    value: Any


@dataclass(frozen=True)
class CandidateIntervention:
    """One auditable, domain-proposed perturbation for Why/Why-not search."""

    candidate_id: str
    factor: str
    description: str
    interventions: tuple[Intervention, ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvironmentSnapshot:
    """Complete state required to restore a simulator frame exactly."""

    environment: str
    frame: int
    state: Any
    environment_rng_state: Any
    policy_rng_state: Any = None
    observations: Mapping[str, Any] = field(default_factory=dict)
    global_state: Any = None
    actions: Mapping[str, str] = field(default_factory=dict)
    action_distributions: Mapping[str, ActionDistribution] = field(default_factory=dict)
    action_masks: Mapping[str, Sequence[float]] = field(default_factory=dict)
    proposed_actions: Mapping[str, str] = field(default_factory=dict)
    executed_actions: Mapping[str, str] = field(default_factory=dict)
    rewards: Mapping[str, float] = field(default_factory=dict)
    policy_hidden_state: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    checkpoint_id: str | None = None


@dataclass(frozen=True)
class RolloutFrame:
    """One decision state and its resulting environment transition.

    ``snapshot`` and ``observations`` are the state on which ``distributions``
    and ``proposed_actions`` were computed. ``next_snapshot`` is the state after
    the environment resolved masks, collisions and other transition rules.
    Keeping both prevents a common XAI error: attaching pre-transition logits to
    post-transition observations.
    """

    frame: int
    snapshot: EnvironmentSnapshot
    observations: Mapping[str, Any]
    actions: Mapping[str, str]
    distributions: Mapping[str, ActionDistribution]
    reward: Mapping[str, float]
    done: bool
    info: Mapping[str, Any]
    proposed_actions: Mapping[str, str] = field(default_factory=dict)
    executed_actions: Mapping[str, str] = field(default_factory=dict)
    action_masks: Mapping[str, Sequence[float]] = field(default_factory=dict)
    next_snapshot: EnvironmentSnapshot | None = None
    reward_breakdown: Mapping[str, float] = field(default_factory=dict)
    task_state: Mapping[str, Any] = field(default_factory=dict)
    charging_state: Mapping[str, Any] = field(default_factory=dict)
    environment_events: tuple[Mapping[str, Any], ...] = ()
    rng_state: Mapping[str, Any] = field(default_factory=dict)
    checkpoint_id: str | None = None


@dataclass(frozen=True)
class RolloutResult:
    """A branch rollout and aggregate action statistics."""

    frames: tuple[RolloutFrame, ...]
    action_frequencies: Mapping[str, Mapping[str, float]]
    terminal_reason: str | None = None


@dataclass(frozen=True)
class EvidenceFact:
    """One environment fact with multilingual text used for independent grounding."""

    fact_id: str
    predicate: str
    arguments: tuple[str, ...]
    value: Any
    factor_groups: tuple[str, ...]
    verbalizations: tuple[str, ...]
    value_verbalizations: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticPolicyContext:
    """Observable semantic inputs and their explanation-time bindings.

    ``features`` is the only field consumed by an extracted policy.  The
    remaining fields preserve where those values came from and which concrete
    entities occupied relational roles at the decision frame.  Keeping those
    concerns separate prevents entity identifiers or post-transition facts
    from leaking into a domain-neutral policy program.
    """

    features: Mapping[str, float]
    entity_bindings: Mapping[str, str] = field(default_factory=dict)
    feature_provenance: Mapping[str, str] = field(default_factory=dict)
    scenario_tags: tuple[str, ...] = ("ordinary",)
    action_constraint_reasons: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict
    )


class PolicyProtocol(Protocol):
    """Minimal policy interface required by adapters and explainers."""

    @property
    def action_names(self) -> tuple[str, ...]: ...

    def act(
        self,
        observations: Mapping[str, Any],
        global_state: Any,
        *,
        deterministic: bool = False,
        fixed_actions: Mapping[str, str] | None = None,
    ) -> tuple[Mapping[str, str], Mapping[str, ActionDistribution]]: ...

    def get_rng_state(self) -> Any: ...

    def set_rng_state(self, state: Any) -> None: ...


class EnvironmentAdapter(ABC):
    """The only environment dependency visible to generic XAI modules."""

    @abstractmethod
    def observation_schema(self) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def action_schema(self) -> Sequence[str]:
        raise NotImplementedError

    @abstractmethod
    def entity_schema(self) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def snapshot(self, policy: PolicyProtocol | None = None) -> EnvironmentSnapshot:
        raise NotImplementedError

    @abstractmethod
    def restore(self, snapshot: EnvironmentSnapshot, policy: PolicyProtocol | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def validate_intervention(
        self,
        snapshot: EnvironmentSnapshot,
        interventions: Sequence[Intervention],
    ) -> tuple[bool, tuple[str, ...]]:
        raise NotImplementedError

    @abstractmethod
    def apply_interventions(
        self,
        snapshot: EnvironmentSnapshot,
        interventions: Sequence[Intervention],
    ) -> EnvironmentSnapshot:
        raise NotImplementedError

    @abstractmethod
    def recompute_observations(self) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def global_state(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def rollout(
        self,
        policy: PolicyProtocol,
        *,
        horizon: int,
        deterministic: bool = False,
    ) -> RolloutResult:
        raise NotImplementedError

    @abstractmethod
    def render(self, state: Any | None = None) -> Any:
        raise NotImplementedError

    @abstractmethod
    def claim_ontology(self) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def policy_entity_references(self) -> Mapping[str, Sequence[str]]:
        """Return canonical policy-controlled entities and multilingual references."""
        raise NotImplementedError

    @abstractmethod
    def policy_entity_properties(self) -> Mapping[str, Any]:
        """Return editable properties for policy-controlled entities."""
        raise NotImplementedError

    @abstractmethod
    def canonicalize_claim_constraint(
        self,
        claim: Any,
        matched_fact: EvidenceFact,
        semantic_matcher: Any,
    ) -> Mapping[str, Any] | None:
        """Translate a matched natural-language claim into a domain constraint.

        This deliberately lives on the adapter: generic grounding and objective-
        validity code must never know environment predicates or value domains.
        """
        raise NotImplementedError

    @abstractmethod
    def claim_fact_compatible(self, claim: Any, fact: EvidenceFact) -> bool:
        """Return whether a claim type can be grounded by this domain fact."""
        raise NotImplementedError

    @abstractmethod
    def claim_value_consistent(
        self,
        claim: Any,
        fact: EvidenceFact,
        semantic_matcher: Any,
    ) -> bool:
        """Check exact domain values after Transformer semantic alignment."""
        raise NotImplementedError

    @abstractmethod
    def counterfactual_action_fact(
        self,
        snapshot: EnvironmentSnapshot,
        target_entity: str,
        action: str,
        distribution: Mapping[str, float],
        interventions: Sequence[Intervention],
    ) -> EvidenceFact:
        """Verbalize an independently recomputed post-intervention policy output."""
        raise NotImplementedError

    @abstractmethod
    def sample_states_from_constraints(
        self,
        claims: Sequence[Any],
        *,
        count: int,
        seed: int,
        base_snapshot: EnvironmentSnapshot | None = None,
    ) -> Sequence[EnvironmentSnapshot]:
        raise NotImplementedError

    @abstractmethod
    def default_target_entity(self, snapshot: EnvironmentSnapshot) -> str:
        raise NotImplementedError

    @abstractmethod
    def policy_distribution(
        self,
        snapshot: EnvironmentSnapshot,
        target_entity: str,
        policy: PolicyProtocol,
    ) -> Mapping[str, float]:
        raise NotImplementedError

    def policy_outputs(
        self,
        snapshot: EnvironmentSnapshot,
        policy: PolicyProtocol,
    ) -> Mapping[str, ActionDistribution]:
        """Query every decentralized Actor on one coherent restored frame."""

        _, distributions = policy.act(
            snapshot.observations,
            snapshot.global_state,
            deterministic=True,
        )
        return distributions

    @abstractmethod
    def evidence_facts(
        self,
        snapshot: EnvironmentSnapshot,
        target_entity: str,
        policy: PolicyProtocol,
    ) -> Sequence[EvidenceFact]:
        raise NotImplementedError

    def actor_observable_facts(
        self,
        snapshot: EnvironmentSnapshot,
        target_entity: str,
        policy: PolicyProtocol,
    ) -> Sequence[EvidenceFact]:
        """Return current-state facts that may be shown to an Actor baseline.

        The default deliberately keeps only facts tagged as current ``state``
        by the environment adapter.  It excludes post-transition resolution,
        extracted-program, counterfactual, and evaluator evidence.  Adapters
        with a narrower observation model may override this method.
        """

        return tuple(
            fact
            for fact in self.evidence_facts(
                snapshot,
                target_entity,
                policy,
            )
            if "state" in fact.factor_groups
        )

    def decision_objective_facts(
        self,
        snapshot: EnvironmentSnapshot,
        target_entity: str,
        policy: PolicyProtocol,
    ) -> Sequence[EvidenceFact]:
        """Return typed evidence for the objective active at decision time.

        Counterfactual action comparisons are incomplete when they only report
        that an action changed: they must also preserve what the policy was
        trying to accomplish before and after the intervention.  The generic
        XAI pipeline identifies that information by the semantic
        ``objective_reason`` factor group rather than by environment-specific
        objective names.  Adapters may override this method when their task
        representation uses a different internal source.

        Only Actor-observable, pre-decision facts are eligible.  Consequently
        an explanation cannot turn a post-transition outcome into a goal the
        Actor supposedly held while choosing its action.
        """

        return tuple(
            fact
            for fact in self.actor_observable_facts(
                snapshot,
                target_entity,
                policy,
            )
            if "objective_reason" in fact.factor_groups
            and (
                not fact.arguments
                or target_entity
                in {str(value) for value in fact.arguments}
            )
        )

    def neural_baseline_execution_facts(
        self,
        snapshot: EnvironmentSnapshot,
        target_entity: str,
        policy: PolicyProtocol,
    ) -> Sequence[EvidenceFact]:
        """Return recorded action and arbitration facts for the NN baseline.

        A neural-output explanation must distinguish the Actor's proposal from
        the action that the environment ultimately executed.  These facts are
        post-decision evidence from the recorded transition; they are not
        Actor inputs and must never be presented as information the Actor knew
        before choosing.  Program traces, interventions, counterfactual
        rollouts, and evaluator verdicts remain outside this baseline.
        """

        relevant_groups = {"action", "action_reason"}
        return tuple(
            fact
            for fact in self.evidence_facts(
                snapshot,
                target_entity,
                policy,
            )
            if relevant_groups.intersection(fact.factor_groups)
            and (
                not fact.arguments
                or target_entity in {str(value) for value in fact.arguments}
            )
        )

    @abstractmethod
    def factor_group_descriptions(self) -> Mapping[str, Sequence[str]]:
        raise NotImplementedError

    @abstractmethod
    def semantic_policy_features(
        self,
        snapshot: EnvironmentSnapshot,
        target_entity: str,
    ) -> Mapping[str, float]:
        """Return named, human-auditable state features for policy extraction."""
        raise NotImplementedError

    def semantic_policy_context(
        self,
        snapshot: EnvironmentSnapshot,
        target_entity: str,
    ) -> SemanticPolicyContext:
        """Return semantic features plus optional relational audit metadata.

        Existing adapters remain compatible: their feature mapping is treated
        as an ordinary context until they opt into richer role bindings and
        action-constraint reasons.
        """

        return SemanticPolicyContext(
            features=dict(
                self.semantic_policy_features(snapshot, target_entity)
            )
        )

    def semantic_observability_audit(
        self,
        first: EnvironmentSnapshot,
        second: EnvironmentSnapshot,
        target_entity: str,
    ) -> Mapping[str, Any]:
        """Check that equal Actor inputs cannot reveal different tree facts."""

        def plain(value: Any) -> Any:
            return value.tolist() if hasattr(value, "tolist") else value

        observations_equal = plain(first.observations[target_entity]) == plain(
            second.observations[target_entity]
        )
        left = self.semantic_policy_context(first, target_entity).features
        right = self.semantic_policy_context(second, target_entity).features
        differences = tuple(
            sorted(
                name
                for name in set(left) | set(right)
                if float(left.get(name, 0.0))
                != float(right.get(name, 0.0))
            )
        )
        return {
            "observations_equal": observations_equal,
            "feature_differences": differences,
            "passed": not observations_equal or not differences,
        }

    def semantic_feature_entity_bindings(
        self,
        feature: str,
        context: SemanticPolicyContext,
    ) -> Mapping[str, str]:
        """Return concrete entities relevant to one domain-owned feature."""

        return {}

    def neural_baseline_explanation_context(
        self,
        context: SemanticPolicyContext,
        selected_action: str,
        *,
        max_features: int = 48,
    ) -> Sequence[Mapping[str, Any]]:
        """Expose Actor-visible decision context without using a surrogate.

        The neural-output baseline still needs observable evidence that can
        answer *why* at a descriptive level.  This default implementation
        keeps the selected action's semantic features plus compact self,
        objective, resource, and relational context.  Raw local-map cells and
        features belonging exclusively to unselected actions are omitted.

        Adapters remain responsible for the natural-language meaning of each
        feature.  No extracted-program path, transition result, intervention,
        or rollout is consulted here.
        """

        action_names = {str(value) for value in self.action_schema()}
        selected = str(selected_action)
        descriptions = self.semantic_feature_descriptions()
        ranked: list[tuple[int, str, Mapping[str, Any]]] = []
        for feature, raw_value in context.features.items():
            feature_name = str(feature)
            components = set(feature_name.split("."))
            mentioned_actions = components.intersection(action_names)
            if mentioned_actions and selected not in mentioned_actions:
                continue
            provenance = str(
                context.feature_provenance.get(feature_name, "observation")
            )
            if provenance == "local_patch":
                continue
            try:
                observed = dict(
                    self.semantic_feature_observation(
                        feature_name,
                        float(raw_value),
                    )
                )
            except (TypeError, ValueError):
                continue
            bindings = dict(
                self.semantic_feature_entity_bindings(feature_name, context)
            )
            record = {
                "evidence_id": f"actor_feature::{feature_name}",
                "feature": feature_name,
                "value": float(raw_value),
                "description": dict(descriptions.get(feature_name, {})),
                "observed_meaning": observed,
                "bound_entities": bindings,
                "provenance": provenance,
            }
            if selected in mentioned_actions:
                priority = 0
            elif bindings:
                priority = 1
            elif provenance not in {"self", "observation"}:
                priority = 2
            else:
                priority = 3
            ranked.append((priority, feature_name, record))
        ranked.sort(key=lambda item: (item[0], item[1]))
        return tuple(
            record
            for _priority, _name, record in ranked[: max(1, int(max_features))]
        )

    @abstractmethod
    def semantic_feature_descriptions(self) -> Mapping[str, Mapping[str, str]]:
        """Return multilingual labels for semantic policy features."""
        raise NotImplementedError

    @abstractmethod
    def semantic_feature_observation(
        self,
        feature: str,
        value: float,
    ) -> Mapping[str, Any]:
        """Verbalize one computed feature value without selecting an explanation.

        This is evidence serialization owned by the environment adapter. It
        prevents a language model from having to guess the sign convention or
        unit of a numerical policy feature.
        """
        raise NotImplementedError

    @abstractmethod
    def action_descriptions(self) -> Mapping[str, Mapping[str, str]]:
        """Return multilingual user-facing action names."""
        raise NotImplementedError

    def objective_descriptions(self) -> Mapping[str, Mapping[str, str]]:
        """Return multilingual names for environment-owned objectives.

        These are vocabulary entries rather than authored explanations.  The
        generic explanation compiler uses them exactly as it uses action
        names, so a typed objective can be realized consistently in every
        supported language without embedding task-specific sentences in the
        XAI core.
        """

        return {}

    def question_vocabulary(self) -> Mapping[str, Any]:
        """Declare the semantic values that free-form questions may target.

        This is an ontology, not a collection of question templates.  The
        Transformer receives canonical IDs plus short multilingual aliases and
        returns one typed target variable.  Environments can extend the state
        variables without changing the shared question parser.
        """

        return {
            "query_variables": {
                "observed_action": {
                    "kind": "action",
                    "aliases": (
                        "current action",
                        "observed action",
                        "当前动作",
                        "这一步做什么",
                    ),
                },
                "last_action": {
                    "kind": "action",
                    "aliases": (
                        "recorded action",
                        "last action",
                        "当前记录动作",
                        "当前移动",
                    ),
                },
                "next_action": {
                    "kind": "action",
                    "aliases": (
                        "next action",
                        "下一步动作",
                        "接下来做什么",
                    ),
                },
                "objective": {
                    "kind": "objective",
                    "aliases": (
                        "current objective",
                        "task goal",
                        "task",
                        "goal",
                        "当前目标",
                        "当前任务",
                        "目标",
                        "任务",
                    ),
                },
            },
            "objectives": {
                str(objective): {
                    **dict(labels),
                    "aliases": tuple(
                        dict.fromkeys(
                            (
                                str(objective),
                                *(
                                    str(value)
                                    for value in labels.values()
                                    if str(value).strip()
                                ),
                            )
                        )
                    ),
                }
                for objective, labels in self.objective_descriptions().items()
            },
            "action_values": {
                str(action): {
                    **dict(labels),
                    "aliases": tuple(
                        dict.fromkeys(
                            (
                                str(action),
                                *(
                                    str(value)
                                    for value in labels.values()
                                    if str(value).strip()
                                ),
                            )
                        )
                    ),
                }
                for action, labels in self.action_descriptions().items()
            },
        }

    def explanation_entity_label(self, entity_id: str, language: str) -> str:
        """Return a short entity label for the environment-neutral explainer.

        Adapters may override this for natural labels.  The default is a pure
        vocabulary fallback and contains no assumption about entity kinds.
        """

        del language
        return str(entity_id).replace("_", " ")

    def explanation_action_label(self, action: str, language: str) -> str:
        """Return an action label without authoring an explanation."""

        labels = self.action_descriptions().get(str(action), {})
        key = "zh" if str(language) == "zh-CN" else "en"
        return str(labels.get(key, action)).strip() or str(action)

    def explanation_objective_label(self, objective: str, language: str) -> str:
        """Return an objective label without encoding a decision rule."""

        labels = self.objective_descriptions().get(str(objective), {})
        key = "zh" if str(language) == "zh-CN" else "en"
        return str(labels.get(key, objective)).strip() or str(objective)

    def explanation_predicate_schema(self) -> Mapping[str, Any]:
        """Declare typed facts that the adapter can expose to ExplanationIR.

        Values describe argument roles and units only.  They must not contain
        a complete answer or a rule deciding which action is preferable.
        """

        return {}

    def explanation_verbalize_unit(
        self,
        unit: Mapping[str, Any],
        language: str,
    ) -> str:
        """Produce a short phrase for one typed explanation unit.

        This generic fallback intentionally exposes only predicate, arguments,
        and value.  A rich adapter can provide friendlier phrases while the XAI
        core remains unaware of environment actions, objectives, or rules.
        """

        arguments = " ".join(str(value) for value in unit.get("arguments", ()))
        predicate = str(unit.get("predicate", "fact")).replace("_", " ")
        value = str(unit.get("value", ""))
        if str(language) == "zh-CN":
            return f"{arguments}的{predicate}为{value}"
        return f"{arguments} has {predicate}: {value}"

    @abstractmethod
    def action_legality_features(self) -> Mapping[str, str]:
        """Map each action to a semantic binary feature describing physical legality."""
        raise NotImplementedError

    def action_constraint_reason_features(
        self,
    ) -> Mapping[str, Mapping[str, str]]:
        """Map actions to observable binary features explaining constraints.

        The default keeps older adapters compatible.  Rich adapters can expose
        domain-owned reason names without teaching the RCPD core what those
        names mean.
        """

        return {}

    def relation_role_definitions(self) -> Mapping[str, str]:
        """Describe dynamic entity roles used by semantic program features."""

        return {}

    @abstractmethod
    def sample_local_policy_states(
        self,
        snapshot: EnvironmentSnapshot,
        target_entity: str,
        *,
        count: int,
        seed: int,
    ) -> Sequence[EnvironmentSnapshot]:
        """Sample complete valid states in the local policy neighborhood."""
        raise NotImplementedError

    def compile_relational_constraints(
        self,
        snapshot: EnvironmentSnapshot,
        constraints: Sequence[Mapping[str, Any]],
    ) -> tuple[Sequence[Intervention], tuple[str, ...]]:
        """Compile domain relations into primitive edits.

        Adapters that do not implement relational scene editing fail explicitly
        instead of silently ignoring a user constraint.
        """

        if not constraints:
            return (), ()
        return (), ("This environment adapter does not support relational constraints.",)

    def refresh_snapshot(
        self,
        snapshot: EnvironmentSnapshot,
    ) -> EnvironmentSnapshot:
        """Recompute actor observations after an edit.

        Domain adapters should override this when stored observations can become
        stale after modifying the underlying state.
        """

        return snapshot

    def causal_intervention_candidates(
        self,
        snapshot: EnvironmentSnapshot,
        target_entity: str,
    ) -> Sequence[CandidateIntervention]:
        """Return legal-domain candidate factors for execution-based Why search.

        The generic simulation layer ranks these candidates using the original
        policy. Adapters provide only editable domain semantics; they do not
        assert that any candidate is actually causal.
        """

        del snapshot, target_entity
        return ()

    def recourse_intervention_candidates(
        self,
        snapshot: EnvironmentSnapshot,
        target_entity: str,
        desired_action: str,
    ) -> Sequence[CandidateIntervention]:
        """Return candidates for minimal legal Why-not recourse search."""

        del desired_action
        return self.causal_intervention_candidates(snapshot, target_entity)

    def sample_evaluation_states(
        self,
        mode: str,
        snapshot: EnvironmentSnapshot,
        target_entity: str,
        *,
        count: int,
        seed: int,
    ) -> Sequence[EnvironmentSnapshot]:
        """Sample mode-specific valid states without leaking domain logic upward."""

        del mode
        return self.sample_local_policy_states(
            snapshot,
            target_entity,
            count=count,
            seed=seed,
        )
