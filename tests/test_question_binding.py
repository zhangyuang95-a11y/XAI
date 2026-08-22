from __future__ import annotations

from typing import Any, Mapping

from backend.nlp.semantic_query_planner import TransformerQueryPlanner
from backend.nlp.tokenizer import CallableTransformerBackend


def _planner(payload: Mapping[str, Any]) -> tuple[TransformerQueryPlanner, list[str]]:
    prompts: list[str] = []

    def generate(prompt: str, schema: str) -> Mapping[str, Any]:
        assert schema == "QuestionIRV2"
        prompts.append(prompt)
        return payload

    return TransformerQueryPlanner(CallableTransformerBackend(generate)), prompts


def _schema(*, focus: str | None = None) -> dict[str, Any]:
    return {
        "actions": ["MOVE", "WAIT"],
        "entities": {
            "agent": {
                "ids": ["robot_1", "robot_2", "robot_3", "robot_4"],
                "references": {
                    "robot_1": ["robot_1", "robot 1", "机器人1"],
                    "robot_2": ["robot_2", "robot 2", "机器人2"],
                    "robot_3": ["robot_3", "robot 3", "机器人3"],
                    "robot_4": ["robot_4", "robot 4", "机器人4"],
                },
            }
        },
        "focus_entity": focus,
    }


def test_unicode_binding_preserves_spaced_chinese_source_span() -> None:
    planner, prompts = _planner(
        {
            "i": "explanatory",
            "t": "robot_2",
            "x": [],
            "r": [],
            "v": ["robot_2.last_action"],
            "d": {},
            "a": [],
        }
    )
    plan = planner.parse(
        "为什么机器人 2 现在选择等待？",
        selected_frame=9,
        environment_schema=_schema(focus="robot_1"),
    )

    assert plan.primary_prediction_target == "robot_2"
    assert plan.entity_roles[0].source_span == "机器人 2"
    assert plan.requires_scene_edit is False
    assert len(prompts) == 1
    assert '"entity_id":"robot_2","source_span":"机器人 2"' in prompts[0]
    assert planner.last_diagnostics["binding"]["target_origin"] == (
        "model_target_confirmed_by_source"
    )


def test_implicit_subject_uses_focus_entity_and_retargets_variables() -> None:
    planner, _ = _planner(
        {
            # A model-selected schema ID is not authoritative when the user
            # omitted an entity and the UI supplied an explicit focus.
            "i": "explanatory",
            "t": "robot_1",
            "x": [],
            "r": [],
            "v": ["robot_1.last_action"],
            "d": {},
            "a": [],
        }
    )
    plan = planner.parse(
        "为什么等待？",
        selected_frame=12,
        environment_schema=_schema(focus="robot_4"),
    )

    assert plan.primary_prediction_target == "robot_4"
    assert plan.target_variables == ("robot_4.last_action",)
    assert plan.entity_roles[0].source_span == ""
    assert plan.clarification_required is False
    assert planner.last_diagnostics["binding"]["target_origin"] == (
        "ui_focus_entity"
    )


def test_missing_model_target_is_completed_from_ui_focus() -> None:
    planner, _ = _planner(
        {
            "i": "explanatory",
            # Small instruction models sometimes omit t and w while still
            # returning a valid semantic intent and query variable.
            "q": "current action",
        }
    )

    plan = planner.parse(
        "为什么这一步停下来了？",
        selected_frame=12,
        environment_schema=_schema(focus="robot_4"),
    )

    assert plan.primary_prediction_target == "robot_4"
    assert plan.target_variables == ("robot_4.current action",)
    assert plan.clarification_required is False
    assert planner.last_diagnostics["binding"]["target_origin"] == (
        "ui_focus_entity"
    )


def test_explicit_entity_overrides_different_focus_entity() -> None:
    planner, _ = _planner(
        {
            "i": "explanatory",
            "t": "robot_2",
            "x": [],
            "r": [],
            "v": ["robot_2.observed_action"],
            "d": {},
            "a": [],
        }
    )
    plan = planner.parse(
        "Why is robot 2 moving now?",
        selected_frame=4,
        environment_schema=_schema(focus="robot_1"),
    )

    assert plan.primary_prediction_target == "robot_2"
    assert plan.entity_roles[0].source_span == "robot 2"


def test_binding_is_environment_neutral_for_hyphenated_aliases() -> None:
    planner, _ = _planner(
        {
            "i": "counterfactual",
            "t": "carrier_a",
            "x": [
                {
                    "op": "move_entity",
                    "e": "scout_b",
                    "p": "position",
                    "val": {"x": 4, "y": 9},
                    # The model paraphrased the connective. The binder must
                    # recover provenance from the original entity and values.
                    "span": "relocate scout b onto coordinates (4, 9)",
                }
            ],
            "r": [],
            "v": ["carrier_a.next_action"],
            "d": {},
            "a": [],
        }
    )
    plan = planner.parse(
        "If scout b moved to (4, 9), what would carrier a do?",
        selected_frame=3,
        environment_schema={
            "actions": ["GLIDE", "HOLD"],
            "entities": {
                "vehicle": {
                    "ids": ["carrier_a", "scout_b"],
                    "references": {
                        "carrier_a": ["carrier-a", "carrier a"],
                        "scout_b": ["scout-b", "scout b"],
                    },
                }
            },
            "focus_entity": "scout_b",
        },
    )

    assert plan.primary_prediction_target == "carrier_a"
    assert plan.intervention_targets == ("scout_b",)
    assert plan.requires_scene_edit is True
    spans = {item.entity_id: item.source_span for item in plan.entity_roles}
    assert spans == {"carrier_a": "carrier a", "scout_b": "scout b"}
    assert plan.interventions[0].source_span == "scout b moved to (4, 9)"


def test_single_explicit_entity_rejects_unmentioned_model_identity() -> None:
    planner, _ = _planner(
        {
            "i": "explanatory",
            "t": "robot_3",
            "x": [],
            "r": [],
            "v": ["robot_3.last_action"],
            "d": {},
            "a": [],
        }
    )
    plan = planner.parse(
        "机器人 2 为什么这样走？",
        selected_frame=6,
        environment_schema=_schema(focus="robot_1"),
    )

    assert plan.primary_prediction_target == "robot_2"
    assert plan.target_variables == ("robot_2.last_action",)
    assert planner.last_diagnostics["binding"]["target_origin"] == (
        "single_explicit_entity"
    )
