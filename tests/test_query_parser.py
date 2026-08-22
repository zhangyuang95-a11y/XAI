from __future__ import annotations

import pytest

from backend.nlp.semantic_query_planner import (
    SemanticTransformerQueryPlanner as TransformerQueryPlanner,
)
from backend.nlp.explanation_generator import (
    ExecutionGroundedExplanationGenerator,
    _output_language_instruction,
)
from backend.nlp.schemas import PrimitiveOperation, QueryIntent
from backend.nlp.tokenizer import CallableTransformerBackend


@pytest.mark.parametrize(
    ("question", "source_span", "language"),
    (
        (
            "假如此时机器人2电量为10%，会怎么走？",
            "机器人2电量为10%",
            "zh-CN",
        ),
        (
            "If robot 2 had 10% battery now, how would it move?",
            "robot 2 had 10% battery",
            "en",
        ),
    ),
)
def test_explicit_battery_hypothesis_cannot_degrade_to_ordinary_query(
    question: str,
    source_span: str,
    language: str,
) -> None:
    """The unified question IR owns roles and edit commitments together."""

    def generate(_prompt: str, schema: str) -> dict[str, object]:
        assert schema == "QuestionIRV2"
        return {
            "intent": "counterfactual",
            "target_entity": "robot_2",
            "referenced_entities": [],
            "entity_roles": [
                {
                    "entity_id": "robot_2",
                    "roles": ["prediction_target", "intervention_subject"],
                    "source_span": source_span,
                }
            ],
            "primitive_interventions": [
                {
                    "operation": "set_battery",
                    "entity_id": "robot_2",
                    "value": 10,
                    "source_span": source_span,
                }
            ],
            "relational_constraints": [],
            "target_variables": ["robot_2.next_action"],
            "causal_variables": [],
            "desired_outcomes": {},
            "ambiguities": [],
        }

    plan = TransformerQueryPlanner(
        CallableTransformerBackend(generate),
        verify_response_language=False,
    ).parse(
        question,
        selected_frame=18,
        environment_schema={
            "entities": {
                "agent": {
                    "ids": ("robot_1", "robot_2"),
                    "properties": {
                        "battery": {"type": "percentage"},
                        "position": {"type": "coordinate"},
                    },
                }
            }
        },
    )

    assert plan.intent.value == "counterfactual"
    assert plan.primary_prediction_target == "robot_2"
    assert plan.intervention_targets == ("robot_2",)
    assert plan.requires_scene_edit is True
    assert plan.requires_simulation is True
    assert plan.requires_baseline_comparison is True
    assert plan.scene_edit_plan is not None
    assert len(plan.interventions) == 1
    intervention = plan.interventions[0]
    assert intervention.operation == PrimitiveOperation.SET_BATTERY
    assert intervention.entity_id == "robot_2"
    assert intervention.value == 10


def test_ordinary_action_question_cannot_acquire_a_hallucinated_edit() -> None:
    def generate(_prompt: str, schema: str) -> dict[str, object]:
        assert schema == "QuestionIRV2"
        return {
            "intent": "explanatory",
            "target_entity": "robot_2",
            "referenced_entities": [],
            "entity_roles": [
                {
                    "entity_id": "robot_2",
                    "roles": ["prediction_target"],
                    "source_span": "robot 2",
                }
            ],
            "primitive_interventions": [],
            "relational_constraints": [],
            "target_variables": ["robot_2.observed_action"],
            "causal_variables": [],
            "desired_outcomes": {},
            "ambiguities": [],
        }

    plan = TransformerQueryPlanner(
        CallableTransformerBackend(generate),
        verify_response_language=False,
    ).parse(
        "Why is robot 2 waiting?",
        selected_frame=18,
        environment_schema={
            "entities": {
                "agent": {
                    "ids": ("robot_2",),
                    "properties": {"battery": {"type": "percentage"}},
                }
            }
        },
    )

    assert plan.intent.value == "explanatory"
    assert plan.requires_scene_edit is False
    assert plan.scene_edit_plan is None


def test_compact_current_direction_why_uses_snapshot_action_without_edit() -> None:
    calls = 0
    captured_prompt = ""

    def generate(prompt: str, schema: str) -> dict[str, object]:
        nonlocal calls, captured_prompt
        calls += 1
        captured_prompt = prompt
        assert schema == "QuestionIRV2"
        return {
            "i": "explanatory",
            "t": "robot_2",
            "x": [],
            "r": [],
            "v": ["robot_2.last_action"],
            "d": {},
            "a": [],
        }

    question = "机器人2当前为什么往这个方向移动？"
    plan = TransformerQueryPlanner(
        CallableTransformerBackend(generate),
        verify_response_language=False,
    ).parse(
        question,
        selected_frame=18,
        environment_schema={
            "entities": {
                "agent": {
                    "ids": ("robot_2",),
                    "references": {
                        "robot_2": ("robot_2", "robot 2", "机器人2"),
                    },
                }
            }
        },
    )

    assert calls == 1
    assert "deictic action" in captured_prompt
    assert plan.intent.value == "explanatory"
    assert plan.primary_prediction_target == "robot_2"
    assert plan.frame_reference == 18
    assert plan.target_variables == ("robot_2.last_action",)
    assert plan.requires_scene_edit is False
    assert plan.scene_edit_plan is None


def test_counterfactual_entity_roles_override_mention_order_end_to_end() -> None:
    """The edited entity must never become the prediction target by position."""

    def generate(_prompt: str, schema: str) -> dict[str, object]:
        assert schema == "QuestionIRV2"
        return {
            "intent": "counterfactual",
            "target_entity": "robot_1",
            "referenced_entities": ["robot_4"],
            "entity_roles": [
                {
                    "entity_id": "robot_4",
                    "roles": ["intervention_subject"],
                    "source_span": "把机器人4移动到坐标(5,5)",
                },
                {
                    "entity_id": "robot_1",
                    "roles": ["prediction_target"],
                    "source_span": "机器人1会怎么移动",
                },
            ],
            "primitive_interventions": [
                {
                    "operation": "move_entity",
                    "entity_id": "robot_4",
                    "value": [5, 5],
                    "source_span": "把机器人4移动到坐标(5,5)",
                }
            ],
            "relational_constraints": [],
            "target_variables": ["robot_1.next_action"],
            "causal_variables": [],
            "desired_outcomes": {},
            "ambiguities": [],
            "horizon": 1,
            "rollout_count": 2,
        }

    plan = TransformerQueryPlanner(
        CallableTransformerBackend(generate),
        verify_response_language=False,
    ).parse(
        "如果把机器人4移动到坐标(5,5)，机器人1会怎么移动?",
        selected_frame=18,
        environment_schema={
            "entities": {
                "agent": {
                    "ids": ("robot_1", "robot_2", "robot_3", "robot_4")
                }
            }
        },
    )

    assert plan.primary_prediction_target == "robot_1"
    assert plan.subjects == ("robot_1",)
    assert plan.intervention_targets == ("robot_4",)
    assert plan.target_variables == ("robot_1.next_action",)
    assert plan.interventions[0].entity_id == "robot_4"
    assert not plan.clarification_required


def test_english_counterfactual_roles_are_independent_of_mention_order() -> None:
    def generate(_prompt: str, schema: str) -> dict[str, object]:
        assert schema == "QuestionIRV2"
        return {
            "intent": "counterfactual",
            "target_entity": "robot_1",
            "referenced_entities": ["robot_4"],
            "entity_roles": [
                {
                    "entity_id": "robot_4",
                    "roles": ["intervention_subject"],
                    "source_span": "robot 4 to (5,5)",
                },
                {
                    "entity_id": "robot_1",
                    "roles": ["prediction_target"],
                    "source_span": "robot 1 do",
                },
            ],
            "primitive_interventions": [
                {
                    "operation": "move_entity",
                    "entity_id": "robot_4",
                    "value": [5, 5],
                    "source_span": "robot 4 to (5,5)",
                }
            ],
            "relational_constraints": [],
            "target_variables": ["robot_1.next_action"],
            "causal_variables": [],
            "desired_outcomes": {},
            "ambiguities": [],
        }

    plan = TransformerQueryPlanner(
        CallableTransformerBackend(generate),
        verify_response_language=False,
    ).parse(
        "If I move robot 4 to (5,5), what will robot 1 do?",
        selected_frame=18,
        environment_schema={
            "entities": {
                "agent": {"ids": ("robot_1", "robot_4")}
            }
        },
    )

    assert plan.primary_prediction_target == "robot_1"
    assert plan.intervention_targets == ("robot_4",)
    assert plan.target_variables == ("robot_1.next_action",)


def test_one_entity_can_be_both_intervention_and_prediction_target() -> None:
    def generate(_prompt: str, schema: str) -> dict[str, object]:
        if schema == "EntityRoleResolution":
            return {
                "bindings": [
                    {
                        "entity_id": "robot_4",
                        "roles": [
                            "intervention_subject",
                            "prediction_target",
                        ],
                        "source_span": "机器人4移动到坐标(5,5)后会怎么走",
                    }
                ],
                "ambiguities": [],
            }
        return {
            "intent": "counterfactual",
            "subjects": ["robot_2"],
            "referenced_entities": [],
            "primitive_interventions": [
                {
                    "operation": "move_entity",
                    "entity_id": "robot_4",
                    "value": [5, 5],
                    "source_span": "机器人4移动到坐标(5,5)",
                }
            ],
            "relational_constraints": [],
            "target_variables": ["robot_4.next_action"],
            "causal_variables": [],
        }

    plan = TransformerQueryPlanner(
        CallableTransformerBackend(generate),
        verify_response_language=False,
    ).parse(
        "机器人4移动到坐标(5,5)后会怎么走?",
        selected_frame=18,
        environment_schema={
            "entities": {"agent": {"ids": ("robot_4",)}}
        },
    )

    assert plan.primary_prediction_target == "robot_4"
    assert plan.intervention_targets == ("robot_4",)


def test_transformer_planner_supports_compositional_relational_edits() -> None:
    backend = CallableTransformerBackend(
        lambda _prompt, _schema: {
            "intent": "counterfactual",
            "subjects": ["robot_1"],
            "referenced_entities": ["robot_1", "robot_2", "robot_3", "robot_4"],
            "requires_scene_edit": True,
            "requires_simulation": True,
            "requires_baseline_comparison": True,
            "scene_edit_plan": {
                "source_frame": 3,
                "relational_constraints": [
                    {
                        "relation": "surrounds",
                        "subjects": ["robot_1", "robot_2", "robot_3", "robot_4"],
                        "source_span": "all other robots surround me",
                    }
                ],
                "simulation_horizon": 4,
                "confidence": 0.99,
            },
            "horizon": 4,
            "rollout_count": 5,
            "confidence": 0.99,
        }
    )

    plan = TransformerQueryPlanner(backend).parse(
        "If all other robots surround me, what will I do?",
        selected_frame=3,
    )

    assert plan.requires_scene_edit
    assert plan.scene_edit_plan is not None
    assert plan.scene_edit_plan.relational_constraints[0].relation.value == "surrounds"
    assert plan.rollout_count == 5


def test_invalid_counterfactual_ir_does_not_trigger_model_repair() -> None:
    semantic_calls = 0

    def generate(_prompt: str, schema: str) -> dict[str, object]:
        nonlocal semantic_calls
        assert schema == "QuestionIRV2"
        semantic_calls += 1
        return {
            "intent": "counterfactual",
            "target_entity": "robot_3",
            "referenced_entities": ["robot_1"],
            "entity_roles": [
                {
                    "entity_id": "robot_3",
                    "roles": ["prediction_target"],
                    "source_span": "机器人3会怎么移动",
                }
            ],
            "primitive_interventions": [],
            "relational_constraints": [],
            "target_variables": ["robot_3.next_action"],
            "desired_outcomes": {},
            "ambiguities": [],
        }

    plan = TransformerQueryPlanner(
        CallableTransformerBackend(generate),
        verify_response_language=False,
    ).parse(
        "如果把机器人1移动到坐标(5,5)，机器人3会怎么移动？",
        selected_frame=5,
        environment_schema={
            "entities": {"robot_1": {}, "robot_3": {}},
        },
    )

    assert plan.clarification_required
    assert not plan.requires_scene_edit
    assert semantic_calls == 1


def test_planner_exposes_top_level_interventions_constraints_and_why_not_goal() -> None:
    backend = CallableTransformerBackend(
        lambda _prompt, _schema: {
            "intent": "why_not",
            "subjects": ["robot_3"],
            "referenced_entities": ["robot_2", "robot_3"],
            "requires_scene_edit": True,
            "requires_simulation": True,
            "requires_baseline_comparison": True,
            "interventions": [
                {
                    "operation": "batch_intervention",
                    "entity_id": "batch",
                    "value": [
                        {
                            "operation": "move_entity",
                            "entity_id": "robot_2",
                            "value": [5, 8],
                        },
                        {
                            "operation": "set_battery",
                            "entity_id": "robot_3",
                            "value": 5,
                        },
                    ],
                }
            ],
            "scene_constraints": [
                {
                    "relation": "adjacent",
                    "subjects": ["robot_2", "robot_3"],
                }
            ],
            "desired_outcomes": {
                "robot_3.next_action": "LEFT"
            },
            "horizon": 3,
            "rollout_count": 4,
            "confidence": 0.98,
        }
    )
    plan = TransformerQueryPlanner(backend).parse(
        "Why did robot 3 not move left after these changes?",
        selected_frame=11,
    )

    assert plan.frame_reference == 11
    assert plan.interventions[0].operation == (
        PrimitiveOperation.BATCH_INTERVENTION
    )
    assert len(plan.interventions[0].value) == 2
    assert plan.scene_constraints[0].relation.value == "adjacent"
    assert plan.desired_outcomes["robot_3.next_action"] == "LEFT"


def test_low_confidence_does_not_block_a_valid_explanation_plan() -> None:
    backend = CallableTransformerBackend(
        lambda _prompt, _schema: {
            "intent": "explanatory",
            "subjects": ["robot_3"],
            "requires_policy_query": True,
            "requires_program_trace": True,
            "target_variables": ["robot_3.observed_action"],
            "evidence_requirements": [
                "state",
                "neural_policy",
                "program_trace",
            ],
            "confidence": 0.2,
        }
    )
    plan = TransformerQueryPlanner(backend).parse(
        "Why is robot 3 waiting?",
        selected_frame=4,
    )

    assert not plan.clarification_required
    # Model self-confidence is intentionally not part of the semantic IR and
    # cannot gate a valid request.
    assert plan.confidence == 0.0
    assert plan.subjects == ("robot_3",)


def test_missing_transformer_backend_fails() -> None:
    with pytest.raises(ValueError, match="Transformer backend"):
        TransformerQueryPlanner(None)  # type: ignore[arg-type]


def test_explanatory_plan_repairs_inconsistent_execution_flags() -> None:
    backend = CallableTransformerBackend(
        lambda _prompt, _schema: {
            "intent": "explanatory",
            "subjects": ["robot_2"],
            "requires_policy_query": False,
            "requires_program_trace": False,
            "target_variables": ["robot_2.observed_action"],
            "rollout_count": 1,
            "evidence_requirements": ["state"],
            "response_language": "zh-CN",
            "confidence": 0.95,
        }
    )

    plan = TransformerQueryPlanner(backend).parse(
        "为什么此时机器人2要左转？",
        selected_frame=115,
    )

    assert plan.requires_policy_query
    assert plan.requires_program_trace
    assert plan.rollout_count == 2
    assert "neural_policy" in plan.evidence_requirements
    assert "causal_analysis" in plan.evidence_requirements
    assert plan.response_language == "zh-CN"


def test_invalid_hallucinated_edit_requests_clarification_without_repair() -> None:
    backend = CallableTransformerBackend(
        lambda _prompt, _schema: {
            "intent": "explanatory",
            "subjects": ["robot_4"],
            "requires_scene_edit": True,
            "requires_simulation": True,
            "requires_baseline_comparison": True,
            "requires_program_trace": True,
            "scene_edit_plan": {
                "source_frame": 10,
                "primitive_interventions": [
                    {
                        "operation": "set_attribute",
                        "entity_id": "robot_4",
                        # Missing property_name: structurally invalid.
                        "value": "WAIT",
                    }
                ],
            },
            "target_variables": ["robot_4.observed_action"],
            "horizon": 1,
            "confidence": 0.95,
        }
    )

    plan = TransformerQueryPlanner(
        backend,
        verify_response_language=False,
    ).parse("Why is robot 4 waiting here?", selected_frame=10)

    assert plan.scene_edit_plan is None
    assert plan.requires_scene_edit is False
    assert plan.requires_simulation is False
    assert plan.requires_baseline_comparison is False
    assert plan.requires_program_trace is False
    assert plan.clarification_required is True
    assert plan.unsupported_components == ()


def test_non_action_why_not_is_not_silently_rewritten_as_an_action_question() -> None:
    backend = CallableTransformerBackend(
        lambda _prompt, _schema: {
            "intent": "why_not",
            "subjects": ["robot_4"],
            "requires_program_trace": True,
            "requires_simulation": True,
            "requires_baseline_comparison": True,
            "target_variables": [
                "robot_2.active",
                "robot_2.carrying_shared_task",
            ],
            "desired_outcomes": {
                "robot_2.active": False,
                "robot_2.carrying_shared_task": True,
            },
            "horizon": 1,
            "confidence": 0.95,
        }
    )

    plan = TransformerQueryPlanner(
        backend,
        verify_response_language=False,
    ).parse("Why is robot 4 waiting here?", selected_frame=10)

    assert plan.intent.value == "why_not"
    assert plan.desired_outcomes == {
        "robot_2.active": False,
        "robot_2.carrying_shared_task": True,
    }
    assert plan.requires_simulation is False
    assert plan.requires_baseline_comparison is False
    assert plan.requires_program_trace is False


def test_planner_preserves_a_why_not_objective_and_adapter_vocabulary() -> None:
    seen_prompt = ""

    def generate(prompt: str, schema: str) -> dict[str, object]:
        nonlocal seen_prompt
        assert schema == "QuestionIRV2"
        seen_prompt = prompt
        return {
            # The model extracted the objective alternative correctly but used
            # the broader explanatory label.  The compiler enforces the
            # desired-outcome => why_not invariant.
            "i": "explanatory",
            "t": "robot_1",
            "q": "objective",
            "w": "charge",
            "x": [],
            "r": [],
            "a": [],
        }

    plan = TransformerQueryPlanner(
        CallableTransformerBackend(generate),
        verify_response_language=False,
    ).parse(
        "为什么机器人1当前不去充电？",
        selected_frame=18,
        environment_schema={
            "entities": {
                "agent": {
                    "ids": ("robot_1",),
                    "references": {"robot_1": ("机器人1",)},
                }
            },
            "actions": ("UP", "WAIT"),
            "query_variables": {
                "observed_action": {"kind": "action"},
                "objective": {"kind": "objective"},
            },
            "objectives": {
                "charge": {"zh": "充电", "aliases": ("充电",)},
                "delivery": {"zh": "送货", "aliases": ("送货",)},
            },
        },
    )

    assert plan.intent == QueryIntent.WHY_NOT
    assert plan.target_variables == ("robot_1.objective",)
    assert plan.desired_outcomes == {"robot_1.objective": "charge"}
    assert not plan.requires_program_trace
    assert not plan.requires_simulation
    assert '"charge"' in seen_prompt
    assert '"objective"' in seen_prompt


def test_contrastive_why_not_grounds_the_negated_action_from_source() -> None:
    """Two named actions must not make a natural contrast look ambiguous."""

    planner = TransformerQueryPlanner(
        CallableTransformerBackend(
            lambda _prompt, _schema: {
                # This is the real Qwen failure mode: broad why-not intent is
                # correct, but q/w are copied from the wrong ontology domain.
                "i": "why_not",
                "t": "robot_2",
                "q": "objective",
                "w": "pickup",
            }
        ),
        verify_response_language=False,
    )
    plan = planner.parse(
        "为什么不是向下，而是等待？",
        selected_frame=18,
        environment_schema={
            "entities": {
                "agent": {
                    "ids": ("robot_1", "robot_2"),
                    "references": {
                        "robot_1": ("机器人1",),
                        "robot_2": ("机器人2",),
                    },
                }
            },
            "focus_entity": "robot_2",
            "actions": ("DOWN", "WAIT"),
            "query_variables": {
                "observed_action": {"kind": "action"},
                "objective": {"kind": "objective"},
            },
            "action_values": {
                "DOWN": {"zh": "向下", "aliases": ("向下",)},
                "WAIT": {"zh": "等待", "aliases": ("等待",)},
            },
            "objectives": {
                "pickup": {"zh": "取货", "aliases": ("取货",)},
            },
        },
    )

    assert plan.clarification_required is False
    assert plan.intent == QueryIntent.WHY_NOT
    assert plan.primary_prediction_target == "robot_2"
    assert plan.target_variables == ("robot_2.observed_action",)
    assert plan.desired_outcomes == {"robot_2.observed_action": "DOWN"}


def test_positive_why_action_is_not_rewritten_as_why_not() -> None:
    planner = TransformerQueryPlanner(
        CallableTransformerBackend(
            lambda _prompt, _schema: {
                "i": "explanatory",
                "t": "robot_1",
                "q": "current action",
                "w": "UP",
            }
        ),
        verify_response_language=False,
    )
    plan = planner.parse(
        "机器人1为什么向上走？",
        selected_frame=18,
        environment_schema={
            "entities": {
                "agent": {
                    "ids": ("robot_1",),
                    "references": {"robot_1": ("机器人1",)},
                }
            },
            "actions": ("UP",),
            "query_variables": {
                "observed_action": {
                    "kind": "action",
                    "aliases": ("current action",),
                },
            },
            "action_values": {
                "UP": {"zh": "向上", "aliases": ("向上",)},
            },
        },
    )

    assert plan.intent == QueryIntent.EXPLANATORY
    assert plan.desired_outcomes == {}
    assert plan.clarification_required is False


def test_invalid_model_structure_degrades_to_focused_action_question() -> None:
    calls = 0

    def fail_once(_prompt: str, _schema: str) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise ValueError("invalid generated JSON")

    planner = TransformerQueryPlanner(
        CallableTransformerBackend(fail_once),
        verify_response_language=False,
    )
    plan = planner.parse(
        "为什么这一步会这样走？",
        selected_frame=18,
        environment_schema={
            "entities": {
                "agent": {
                    "ids": ("robot_1", "robot_2"),
                    "references": {
                        "robot_1": ("机器人1",),
                        "robot_2": ("机器人2",),
                    },
                }
            },
            "focus_entity": "robot_2",
            "query_variables": {
                "observed_action": {"kind": "action"},
            },
        },
    )

    assert calls == 1
    assert plan.clarification_required is False
    assert plan.intent == QueryIntent.EXPLANATORY
    assert plan.primary_prediction_target == "robot_2"
    assert plan.target_variables == ("robot_2.observed_action",)
    assert planner.last_diagnostics["binding"][
        "degraded_to_source_grounded_default"
    ] is True


def test_invalid_counterfactual_structure_still_requests_clarification() -> None:
    planner = TransformerQueryPlanner(
        CallableTransformerBackend(
            lambda _prompt, _schema: (_ for _ in ()).throw(
                ValueError("invalid generated JSON")
            )
        ),
        verify_response_language=False,
    )
    plan = planner.parse(
        "如果机器人2的电量变成10%，它会怎么走？",
        selected_frame=18,
        environment_schema={
            "entities": {
                "agent": {
                    "ids": ("robot_2",),
                    "references": {"robot_2": ("机器人2",)},
                }
            },
            "focus_entity": "robot_2",
        },
    )

    assert plan.clarification_required is True
    assert "场景修改" in str(plan.clarification_reason)


@pytest.mark.parametrize("assignment_word", ("为", "有"))
def test_literal_percentage_edit_recovers_when_model_output_is_invalid(
    assignment_word: str,
) -> None:
    calls = 0

    def fail_once(_prompt: str, _schema: str) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise ValueError("invalid generated JSON")

    planner = TransformerQueryPlanner(
        CallableTransformerBackend(fail_once),
        verify_response_language=False,
    )
    plan = planner.parse(
        f"如果机器人2此时电量{assignment_word}100%会怎么走",
        selected_frame=179,
        environment_schema={
            "entities": {
                "agent": {
                    "ids": ("robot_1", "robot_2", "robot_3", "robot_4"),
                    "references": {
                        "robot_1": ("机器人1",),
                        "robot_2": ("机器人2",),
                        "robot_3": ("机器人3",),
                        "robot_4": ("机器人4",),
                    },
                    "properties": {
                        "battery": {
                            "type": "percentage",
                            "texts": ("battery", "电量"),
                        }
                    },
                }
            },
            # The literal entity in the question must override the UI focus.
            "focus_entity": "robot_4",
            "query_variables": {
                "next_action": {"kind": "action"},
            },
        },
    )

    assert calls == 1
    assert plan.clarification_required is False
    assert plan.intent == QueryIntent.COUNTERFACTUAL
    assert plan.primary_prediction_target == "robot_2"
    assert plan.target_variables == ("robot_2.next_action",)
    assert plan.requires_scene_edit is True
    assert len(plan.interventions) == 1
    intervention = plan.interventions[0]
    assert intervention.operation == PrimitiveOperation.SET_ATTRIBUTE
    assert intervention.entity_id == "robot_2"
    assert intervention.property_name == "battery"
    assert intervention.value == 100.0
    assert intervention.source_span == (
        f"机器人2此时电量{assignment_word}100%"
    )
    assert planner.last_diagnostics["binding"][
        "degraded_to_source_grounded_default"
    ] is True


def test_literal_percentage_value_overrides_model_fraction_normalization() -> None:
    planner = TransformerQueryPlanner(
        CallableTransformerBackend(
            lambda _prompt, _schema: {
                "i": "counterfactual",
                "t": "robot_2",
                "q": "next_action",
                "w": None,
                # A common model mistake is converting 100% to the fraction 1.
                "x": [
                    {
                        "op": "set_battery",
                        "e": "robot_2",
                        "val": 1.0,
                        "span": "机器人2此时电量为100%",
                    }
                ],
                "r": [],
                "a": [],
            }
        ),
        verify_response_language=False,
    )
    plan = planner.parse(
        "如果机器人2此时电量为100%会怎么走",
        selected_frame=179,
        environment_schema={
            "entities": {
                "agent": {
                    "ids": ("robot_2", "robot_4"),
                    "references": {
                        "robot_2": ("机器人2",),
                        "robot_4": ("机器人4",),
                    },
                    "properties": {
                        "battery": {
                            "type": "percentage",
                            "texts": ("battery", "电量"),
                        }
                    },
                }
            },
            "focus_entity": "robot_4",
            "query_variables": {
                "next_action": {"kind": "action"},
            },
        },
    )

    assert plan.clarification_required is False
    assert len(plan.interventions) == 1
    assert plan.interventions[0].property_name == "battery"
    assert plan.interventions[0].value == 100.0


@pytest.mark.parametrize("locative", ("在", "再"))
def test_literal_coordinate_edit_recovers_from_environment_position_semantics(
    locative: str,
) -> None:
    calls = 0

    def fail_once(_prompt: str, _schema: str) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise ValueError("invalid generated JSON")

    planner = TransformerQueryPlanner(
        CallableTransformerBackend(fail_once),
        verify_response_language=False,
    )
    plan = planner.parse(
        f"如果此时机器人3{locative}（5，5）会怎么走",
        selected_frame=27,
        environment_schema={
            "entities": {
                "agent": {
                    "ids": ("robot_1", "robot_2", "robot_3", "robot_4"),
                    "references": {
                        "robot_1": ("机器人1",),
                        "robot_2": ("机器人2",),
                        "robot_3": ("机器人3",),
                        "robot_4": ("机器人4",),
                    },
                    "properties": {
                        "position": {
                            "type": "coordinate",
                            "texts": ("position", "location", "位置", "坐标"),
                            "semantic_role": "entity_position",
                            "edit_operation": "move_entity",
                        },
                    },
                }
            },
            "focus_entity": "robot_4",
            "query_variables": {
                "next_action": {"kind": "action"},
            },
        },
    )

    assert calls == 1
    assert plan.clarification_required is False
    assert plan.intent == QueryIntent.COUNTERFACTUAL
    assert plan.primary_prediction_target == "robot_3"
    assert plan.target_variables == ("robot_3.next_action",)
    assert len(plan.interventions) == 1
    intervention = plan.interventions[0]
    assert intervention.operation == PrimitiveOperation.MOVE_ENTITY
    assert intervention.entity_id == "robot_3"
    assert intervention.property_name == "position"
    assert tuple(intervention.value) == (5, 5)
    assert intervention.source_span == f"机器人3{locative}（5，5）"


def test_relation_misplaced_in_query_field_is_grounded_from_literal_alias() -> None:
    planner = TransformerQueryPlanner(
        CallableTransformerBackend(
            lambda _prompt, _schema: {
                "i": "counterfactual",
                "t": "robot_2",
                # Real small-model failure: the relation is understood but
                # emitted as q/w instead of one r item.
                "q": "adjacent",
                "w": "robot_4",
                "x": [],
                "r": [],
                "a": [],
            }
        ),
        verify_response_language=False,
    )
    plan = planner.parse(
        "如果机器人2在机器人4旁边会怎样？",
        selected_frame=18,
        environment_schema={
            "entities": {
                "agent": {
                    "ids": ("robot_1", "robot_2", "robot_4"),
                    "references": {
                        "robot_1": ("机器人1",),
                        "robot_2": ("机器人2",),
                        "robot_4": ("机器人4",),
                    },
                }
            },
            "focus_entity": "robot_2",
            "relations": {
                "adjacent": {
                    "zh": "相邻",
                    "aliases": ("相邻", "旁边"),
                }
            },
        },
    )

    assert plan.clarification_required is False
    assert plan.intent == QueryIntent.COUNTERFACTUAL
    assert plan.requires_scene_edit is True
    assert plan.target_variables == ("robot_2.next_action",)
    assert plan.scene_constraints[0].relation.value == "adjacent"
    assert plan.scene_constraints[0].subjects == ("robot_2", "robot_4")
    assert plan.scene_constraints[0].source_span == "旁边"


def test_planner_maps_unsupported_language_to_english() -> None:
    backend = CallableTransformerBackend(
        lambda _prompt, schema: (
            {
                "input_language": "ja",
                "response_language": "ja",
                "confidence": 0.99,
            }
            if schema == "LanguageIdentification"
            else {
                "intent": "predictive",
                "subjects": ["robot_1"],
                "requires_policy_query": True,
                "target_variables": ["robot_1.next_action"],
                "response_language": "en",
                "confidence": 0.96,
            }
        )
    )

    plan = TransformerQueryPlanner(backend).parse(
        "ロボット1は次にどちらへ動きますか？",
        selected_frame=7,
    )

    assert plan.response_language == "en"
    assert plan.to_dict()["response_language"] == "en"


def test_language_identification_is_local_and_uses_no_second_model_call() -> None:
    schemas: list[str] = []

    def generate(_prompt: str, schema: str) -> dict[str, object]:
        schemas.append(schema)
        if schema == "LanguageIdentification":
            return {
                "input_language": "ar",
                "response_language": "ar",
                "confidence": 0.99,
            }
        return {
            "intent": "explanatory",
            "subjects": ["robot_2"],
            "target_variables": ["robot_2.observed_action"],
            "confidence": 0.95,
        }

    plan = TransformerQueryPlanner(
        CallableTransformerBackend(generate)
    ).parse(
        "لماذا ينعطف الروبوت 2 يسارًا؟",
        selected_frame=12,
    )

    assert schemas == ["QuestionIRV2"]
    assert plan.response_language == "en"


def test_chinese_question_overrides_an_invalid_japanese_language_label() -> None:
    prompts: list[str] = []

    def generate(prompt: str, schema: str) -> dict[str, object]:
        if schema == "LanguageIdentification":
            prompts.append(prompt)
            return {
                "input_language": "ja",
                "response_language": "ja",
                "confidence": 0.99,
            }
        return {
            "intent": "explanatory",
            "subjects": ["robot_2"],
            "target_variables": ["robot_2.observed_action"],
            "confidence": 0.95,
        }

    plan = TransformerQueryPlanner(
        CallableTransformerBackend(generate)
    ).parse(
        "为什么机器人2在这里等待？",
        selected_frame=11,
    )

    assert plan.response_language == "zh-CN"
    assert prompts == []


def test_explanation_language_instruction_is_limited_to_zh_or_en() -> None:
    instruction = _output_language_instruction(
        "ar",
        question="لماذا انعطف الروبوت؟",
    )
    automatic = _output_language_instruction(
        "auto",
        question="¿Por qué giró el robot?",
    )

    assert instruction == "English only"
    assert automatic == "English only"


def test_language_audit_does_not_call_model_repair() -> None:
    backend = CallableTransformerBackend(
        lambda _prompt, schema: (_ for _ in ()).throw(
            AssertionError(f"language audit must not request {schema}")
        ),
        text_generator=lambda _prompt: (
            "机器人2正在等待，让机器人4先通过，从而避免发生碰撞。"
        ),
    )
    generator = ExecutionGroundedExplanationGenerator(backend)

    answer = generator._ensure_requested_output_language(
        "ロボット2はロボット4を先に通すために待っています。",
        requested_language="zh-CN",
        question="为什么机器人2在这里等待？",
    )

    assert answer == "ロボット2はロボット4を先に通すために待っています。"
    assert generator.last_grounding["language_audit"] == {
        "requested": "zh-CN",
        "detected": "zh-CN",
        "rewritten": False,
    }
