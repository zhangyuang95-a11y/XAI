from __future__ import annotations

from evaluation.claim_grounding import (
    ClaimGroundingEvaluator,
    _claim_numeric_literals,
    _structured_value_matches,
    summarize_claim_verdicts,
)
from backend.nlp.schemas import (
    AtomicClaim,
    ClaimVerdictStatus,
    EvidenceBundle,
    QueryIntent,
    QueryPlan,
)
from backend.nlp.tokenizer import CallableTransformerBackend


def _backend() -> CallableTransformerBackend:
    return CallableTransformerBackend(
        lambda _prompt, _schema: {
            "alignments": [
                {
                    "claim_id": "cause",
                    "evidence_ids": ["baseline_0", "counterfactual_0"],
                }
            ]
        }
    )


def test_asserted_value_wrapper_matches_scalar_and_coordinate_records() -> None:
    assert _structured_value_matches(
        {"asserted_value": 3},
        {
            "semantic_name": "remaining_work",
            "value": 3,
        },
    )
    assert _structured_value_matches(
        {"asserted_value": "coordinate_position_6_9"},
        {
            "semantic_name": "selected_destination",
            "value": [6, 9],
        },
    )
    assert _structured_value_matches(
        {"remaining_work": 3},
        {
            "semantic_name": "remaining_work",
            "value": 3,
        },
    )


def test_claim_numbers_ignore_textual_entity_but_keep_coordinate() -> None:
    assert _claim_numeric_literals(
        "机器人2的目的地是(6, 9)",
        ("机器人2", "(6, 9)"),
    ) == (6.0, 9.0)


def _bundle(*, observable: bool, delta: float) -> EvidenceBundle:
    return EvidenceBundle(
        query_plan=QueryPlan("why", QueryIntent.EXPLANATORY, confidence=1.0),
        direct_result={},
        baseline_results=({"distribution": {"WAIT": 0.5}},),
        counterfactual_results=(
            {"action_probability_delta": {"robot_1": {"WAIT": delta}}},
        ),
        policy_results={"causal_observable": observable},
    )


def test_causal_claim_requires_observable_paired_effect() -> None:
    claim = AtomicClaim(
        "cause",
        "robot 2 affected robot 1",
        "causal",
        confidence=1.0,
    )
    evaluator = ClaimGroundingEvaluator(_backend(), causal_effect_threshold=0.01)

    supported = evaluator.evaluate((claim,), _bundle(observable=True, delta=0.2))[0]
    no_effect = evaluator.evaluate((claim,), _bundle(observable=True, delta=0.0))[0]
    hidden = evaluator.evaluate((claim,), _bundle(observable=False, delta=0.2))[0]

    assert supported.status == ClaimVerdictStatus.SUPPORTED
    assert no_effect.status == ClaimVerdictStatus.CONTRADICTED
    assert hidden.status == ClaimVerdictStatus.UNVERIFIABLE


def test_claim_types_select_independent_execution_evidence() -> None:
    alignments = {
        "state": ("state_0",),
        "action": ("neural_policy",),
        "future": ("counterfactual_0",),
        "counterfactual": ("counterfactual_0",),
        "program": ("program_0",),
        "constraint": ("neural_policy",),
        "unknown": (),
    }
    backend = CallableTransformerBackend(
        lambda _prompt, _schema: {
            "alignments": [
                {
                    "claim_id": claim_id,
                    "evidence_ids": list(evidence_ids),
                }
                for claim_id, evidence_ids in alignments.items()
            ]
        }
    )
    plan = QueryPlan(
        "question",
        QueryIntent.MIXED,
        frame_reference=7,
        confidence=1.0,
    )
    bundle = EvidenceBundle(
        query_plan=plan,
        direct_result={},
        state_facts=(
            {
                "predicate": "battery",
                "arguments": ("robot_1",),
                "value": 60.0,
            },
        ),
        counterfactual_results=(
            {
                "terminal_reason": "horizon",
                "action_probability_delta": {
                    "robot_1": {"LEFT": 0.2}
                },
            },
        ),
        policy_results={
            "proposed_action": "LEFT",
            "executed_action": "WAIT",
            "action_mask": (1.0, 1.0, 1.0, 0.0, 1.0),
        },
        program_trace=(
            {
                "branch_id": "b1",
                "feature": "candidate.LEFT.legal",
                "result": True,
            },
        ),
    )
    claims = (
        AtomicClaim(
            "state",
            "battery is 60",
            "state",
            expected_outcome={"value": 60.0},
            confidence=0.9,
        ),
        AtomicClaim(
            "action",
            "actor proposed LEFT",
            "action",
            expected_outcome={"proposed_action": "LEFT"},
            confidence=0.9,
        ),
        AtomicClaim(
            "future",
            "rollout reached horizon",
            "future",
            expected_outcome={"terminal_reason": "horizon"},
            confidence=0.9,
        ),
        AtomicClaim(
            "counterfactual",
            "counterfactual reached horizon",
            "counterfactual",
            expected_outcome={"terminal_reason": "horizon"},
            confidence=0.9,
        ),
        AtomicClaim(
            "program",
            "program branch evaluated true",
            "program",
            expected_outcome={"result": True},
            confidence=0.9,
        ),
        AtomicClaim(
            "constraint",
            "environment executed WAIT",
            "environment_constraint",
            expected_outcome={"executed_action": "WAIT"},
            confidence=0.9,
        ),
        AtomicClaim(
            "unknown",
            "the robot felt worried",
            "mental_state",
            confidence=0.9,
        ),
    )

    verdicts = ClaimGroundingEvaluator(backend).evaluate(
        claims,
        bundle,
    )

    assert [item.status for item in verdicts[:-1]] == [
        ClaimVerdictStatus.SUPPORTED
    ] * 6
    assert verdicts[-1].status == ClaimVerdictStatus.UNVERIFIABLE
    assert all(
        item.evidence[0].frame_id == 7
        for item in verdicts[:2]
    )


def test_empty_transformer_alignment_cannot_hide_direct_action_evidence() -> None:
    backend = CallableTransformerBackend(
        lambda _prompt, _schema: {"alignments": []}
    )
    bundle = EvidenceBundle(
        query_plan=QueryPlan(
            "what action",
            QueryIntent.EXPLANATORY,
            frame_reference=12,
            confidence=1.0,
        ),
        direct_result={},
        policy_results={
            "proposed_action": "LEFT",
            "executed_action": "LEFT",
        },
    )
    claim = AtomicClaim(
        "action",
        "Robot 2 proposed LEFT.",
        "action",
        expected_outcome={"proposed_action": "LEFT"},
        confidence=0.9,
    )

    verdict = ClaimGroundingEvaluator(backend).evaluate(
        (claim,),
        bundle,
    )[0]

    assert verdict.status == ClaimVerdictStatus.SUPPORTED
    assert verdict.evidence[0].evidence_id == "neural_policy"


def test_program_progress_claim_uses_exact_feature_and_numeric_relation() -> None:
    backend = CallableTransformerBackend(
        lambda _prompt, _schema: {
            "alignments": [
                {
                    "claim_id": "progress",
                    "evidence_ids": ["program_1"],
                }
            ]
        }
    )
    bundle = EvidenceBundle(
        query_plan=QueryPlan(
            "why right",
            QueryIntent.EXPLANATORY,
            frame_reference=0,
            confidence=1.0,
        ),
        direct_result={},
        program_trace=(
            {
                "feature": "goal.column_delta",
                "observed_value": 6.0,
            },
            {
                "feature": "candidate.RIGHT.goal_progress",
                "observed_value": 1.0,
            },
        ),
        disagreement={
            "program_available": True,
            "program_reliable": True,
        },
    )
    claim = AtomicClaim(
        "progress",
        "Moving right gets closer to the pickup point.",
        "program",
        predicate="candidate.RIGHT.goal_progress",
        expected_outcome={
            "feature": "candidate.RIGHT.goal_progress",
            "operator": ">",
            "value": 0,
            "exact_value": 1,
        },
        confidence=0.9,
    )

    verdict = ClaimGroundingEvaluator(backend).evaluate(
        (claim,),
        bundle,
    )[0]

    assert verdict.status == ClaimVerdictStatus.SUPPORTED
    assert verdict.evidence[0].evidence_id == "program_1"

    wrong_magnitude = AtomicClaim(
        "progress",
        "Moving right gets two steps closer to the pickup point.",
        "program",
        predicate="candidate.RIGHT.goal_progress",
        expected_outcome={
            "feature": "candidate.RIGHT.goal_progress",
            "operator": ">",
            "value": 0,
            "exact_value": 2,
        },
        confidence=0.9,
    )
    contradicted = ClaimGroundingEvaluator(backend).evaluate(
        (wrong_magnitude,),
        bundle,
    )[0]
    assert contradicted.status == ClaimVerdictStatus.CONTRADICTED


def test_multilingual_goal_reference_matches_recorded_state_fact() -> None:
    backend = CallableTransformerBackend(
        lambda _prompt, _schema: {
            "alignments": [
                {
                    "claim_id": "goal",
                    "evidence_ids": ["state_0"],
                }
            ]
        }
    )
    bundle = EvidenceBundle(
        query_plan=QueryPlan(
            "where",
            QueryIntent.EXPLANATORY,
            frame_reference=0,
            confidence=1.0,
        ),
        direct_result={},
        state_facts=(
            {
                "fact_id": "robot_2.goal_position",
                "predicate": "goal_position",
                "arguments": ("robot_2", "pickup"),
                "value": (7, 10),
                "verbalizations": (
                    "robot_2 pickup goal is at row 7 column 10",
                    "robot_2的个人取货目标位于第7行第10列",
                ),
                "value_verbalizations": (
                    "row 7 column 10",
                    "第7行第10列",
                    "pickup point",
                    "取货点",
                ),
            },
        ),
    )
    claim = AtomicClaim(
        "goal",
        "机器人2正在去取货点",
        "state",
        entities=("机器人2",),
        predicate="goal_position",
        expected_outcome={
            "predicate": "goal_position",
            "arguments": ["机器人2", "取货点"],
        },
        confidence=0.9,
    )

    verdict = ClaimGroundingEvaluator(backend).evaluate(
        (claim,),
        bundle,
    )[0]

    assert verdict.status == ClaimVerdictStatus.SUPPORTED
    assert verdict.evidence[0].evidence_id == "state_0"


def test_invalid_alignment_json_does_not_discard_direct_verification() -> None:
    def fail_alignment(_prompt: str, _schema: str):
        raise ValueError(
            "Transformer returned invalid JSON for ClaimEvidenceAlignment"
        )

    bundle = EvidenceBundle(
        query_plan=QueryPlan(
            "why left",
            QueryIntent.EXPLANATORY,
            frame_reference=12,
            confidence=1.0,
        ),
        direct_result={},
        policy_results={
            "proposed_action": "LEFT",
            "executed_action": "LEFT",
        },
    )
    claim = AtomicClaim(
        "action",
        "Robot 2 proposed LEFT.",
        "action",
        expected_outcome={"proposed_action": "LEFT"},
        confidence=0.9,
    )

    verdict = ClaimGroundingEvaluator(
        CallableTransformerBackend(fail_alignment)
    ).evaluate((claim,), bundle)[0]

    assert verdict.status == ClaimVerdictStatus.SUPPORTED
    assert verdict.evidence[0].evidence_id == "neural_policy"
    assert "alignment was unavailable" in verdict.verifier_reason


def test_environment_supplied_semantic_leaves_are_verified_generically() -> None:
    """The evaluator must not need domain predicates or prose templates."""

    backend = CallableTransformerBackend(
        lambda _prompt, _schema: {
            "alignments": [
                {
                    "claim_id": "energy",
                    "evidence_assertions": [
                        {
                            "evidence_id": "state_0_requirement_0",
                            "claim_value": 24.0,
                        },
                        {
                            "evidence_id": "state_0_requirement_1",
                            "claim_value": 40.0,
                        },
                    ],
                }
            ]
        }
    )
    bundle = EvidenceBundle(
        query_plan=QueryPlan(
            "multilingual natural-language question",
            QueryIntent.EXPLANATORY,
            frame_reference=5,
            confidence=1.0,
        ),
        direct_result={},
        state_facts=(
            {
                "fact_id": "entity_alpha.reason",
                "predicate": "environment_defined_reason",
                "arguments": ("entity_alpha", "objective_beta"),
                "value": {
                    "schema": "environment_reason.v1",
                    "explanation_requirements": [
                        {
                            "key": "reason.current_resource",
                            "semantic_name": "current_resource",
                            "role": "objective_reason",
                            "group": "resource",
                            "value": 24.0,
                            "unit": "environment_units",
                        },
                        {
                            "key": "reason.required_resource",
                            "semantic_name": "required_resource",
                            "role": "objective_reason",
                            "group": "resource",
                            "value": 40.0,
                            "unit": "environment_units",
                            "relation": {
                                "operator": "<",
                                "other_key": "reason.current_resource",
                            },
                        },
                    ],
                },
            },
        ),
    )
    claim = AtomicClaim(
        "energy",
        "The entity has 24 units, below the required 40.",
        "state",
        entities=("entity_alpha",),
        expected_outcome={"resource": 24, "threshold": 40},
        confidence=0.9,
    )

    verdict = ClaimGroundingEvaluator(backend).evaluate(
        (claim,),
        bundle,
    )[0]

    assert verdict.status == ClaimVerdictStatus.SUPPORTED
    assert {
        item.evidence_id for item in verdict.evidence
    } == {
        "state_0_requirement_0",
        "state_0_requirement_1",
    }


def test_semantic_leaf_alignment_does_not_turn_false_claim_true() -> None:
    backend = CallableTransformerBackend(
        lambda _prompt, _schema: {
            "alignments": [
                {
                    "claim_id": "resource",
                    "evidence_assertions": [
                        {
                            "evidence_id": "state_0_requirement_0",
                            "claim_value": 99.0,
                        }
                    ],
                }
            ]
        }
    )
    bundle = EvidenceBundle(
        query_plan=QueryPlan(
            "question",
            QueryIntent.EXPLANATORY,
            confidence=1.0,
        ),
        direct_result={},
        state_facts=(
            {
                "predicate": "arbitrary_reason",
                "arguments": ("entity_alpha",),
                "value": {
                    "explanation_requirements": [
                        {
                            "key": "reason.resource",
                            "semantic_name": "resource",
                            "role": "objective_reason",
                            "group": "resource",
                            "value": 24.0,
                        }
                    ]
                },
            },
        ),
    )
    claim = AtomicClaim(
        "resource",
        "The resource value is 99.",
        "state",
        expected_outcome={"resource": 99},
        confidence=0.9,
    )

    verdict = ClaimGroundingEvaluator(backend).evaluate(
        (claim,),
        bundle,
    )[0]

    assert verdict.status == ClaimVerdictStatus.CONTRADICTED


def test_explanation_metrics_do_not_invent_reference_dependent_scores() -> None:
    claim = AtomicClaim(
        "c1",
        "paired rollout changed the action",
        "counterfactual",
        confidence=0.8,
    )
    verdict = ClaimGroundingEvaluator(_backend()).evaluate(
        (AtomicClaim("cause", "cause", "causal", confidence=0.8),),
        _bundle(observable=True, delta=0.2),
    )[0]
    verdict = type(verdict)(
        claim=claim,
        status=verdict.status,
        evidence=verdict.evidence,
        confidence=verdict.confidence,
        verifier_reason=verdict.verifier_reason,
    )
    without_reference = summarize_claim_verdicts((verdict,))
    with_reference = summarize_claim_verdicts(
        (verdict,),
        reference_statuses={
            "c1": ClaimVerdictStatus.SUPPORTED
        },
        program_disagreement_present=True,
        program_disagreement_detected=True,
    )

    assert without_reference.counterfactual_correctness is None
    assert without_reference.calibration_error is None
    assert with_reference.counterfactual_correctness == 1.0
    assert (
        with_reference.program_actor_disagreement_detection_rate
        == 1.0
    )
    assert with_reference.calibration_error is not None
