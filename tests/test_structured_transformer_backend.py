from __future__ import annotations

from backend.nlp.tokenizer import HuggingFaceStructuredTransformer


def _backend_with_responses(
    *responses: str,
) -> tuple[HuggingFaceStructuredTransformer, list[str]]:
    backend = object.__new__(HuggingFaceStructuredTransformer)
    backend.json_repair_attempts = max(0, len(responses) - 1)
    prompts: list[str] = []
    generated = iter(responses)

    def generate_text(prompt: str) -> str:
        prompts.append(prompt)
        return next(generated)

    backend.generate_text = generate_text  # type: ignore[method-assign]
    return backend, prompts


def test_generate_json_accepts_markdown_and_ignores_duplicate_object() -> None:
    backend, _ = _backend_with_responses(
        'Analysis first.\n```json\n'
        '{"intent":"explanatory","subjects":["robot_2"],"confidence":0.97}'
        '\n```\n{"duplicate":true}'
    )

    payload = backend.generate_json("plan this", schema_name="QueryPlan")

    assert payload["intent"] == "explanatory"
    assert payload["subjects"] == ["robot_2"]


def test_generate_json_accepts_natural_language_explanation_envelope() -> None:
    backend, _ = _backend_with_responses(
        '{"answer":"Robot 1 would move left after the edit."}'
    )

    payload = backend.generate_json(
        "realize this counterfactual",
        schema_name="NaturalLanguageExplanation",
    )

    assert payload == {
        "answer": "Robot 1 would move left after the edit.",
    }


def test_generate_json_retries_when_first_object_has_wrong_schema() -> None:
    backend, prompts = _backend_with_responses(
        '{"message":"not a query plan"}',
        '{"intent":"factual","subjects":["robot_1"],"confidence":0.9}',
    )

    payload = backend.generate_json("plan this", schema_name="QueryPlan")

    assert payload["intent"] == "factual"
    assert len(prompts) == 2
    assert "Previous response" in prompts[1]
    assert '"message":"not a query plan"' in prompts[1]


def test_language_schema_rejects_every_tag_except_chinese_and_english() -> None:
    backend, prompts = _backend_with_responses(
        '{"input_language":"ja","response_language":"ja"}',
        '{"input_language":"zh-CN","response_language":"zh-CN"}',
    )

    payload = backend.generate_json(
        "identify this language",
        schema_name="LanguageIdentification",
    )

    assert payload == {
        "input_language": "zh-CN",
        "response_language": "zh-CN",
    }
    assert len(prompts) == 2


def test_generate_json_unwraps_named_schema_object() -> None:
    backend, _ = _backend_with_responses(
        '{"AtomicClaimList":{"claims":[]}}'
    )

    payload = backend.generate_json(
        "extract claims",
        schema_name="AtomicClaimList",
    )

    assert payload == {"claims": []}


def test_generate_json_recovers_unwrapped_claim_objects() -> None:
    backend, _ = _backend_with_responses(
        '{"claim_id":"claim_1","text":"机器人电量为38%。",'
        '"claim_type":"state","expected_outcome":{"value":38}}'
        '\n'
        '{"claim_id":"claim_2","text":"机器人向右移动。",'
        '"claim_type":"action","expected_outcome":{"action":"RIGHT"}}'
    )

    payload = backend.generate_json(
        "extract claims",
        schema_name="AtomicClaimList",
    )

    assert [item["claim_id"] for item in payload["claims"]] == [
        "claim_1",
        "claim_2",
    ]


def test_generate_json_recovers_values_embedded_in_schema() -> None:
    backend, _ = _backend_with_responses(
        """
        {
          "type": "object",
          "required": ["intent", "subjects"],
          "properties": {
            "intent": {
              "type": "string",
              "enum": ["explanatory", "why_not"]
            },
            "frame_reference": {
              "type": "integer",
              "value": 31
            },
            "subjects": {
              "type": "array",
              "items": [
                {"type": "string", "value": "robot_2"}
              ]
            },
            "requires_program_trace": {
              "type": "boolean",
              "value": true
            },
            "target_variables": {
              "type": "array",
              "items": [
                {
                  "type": "string",
                  "value": "robot_2.observed_action"
                }
              ]
            },
            "desired_outcomes": {
              "type": "object",
              "properties": {
                "robot_2.next_action": {
                  "type": "string",
                  "value": "LEFT"
                }
              }
            },
            "response_language": {
              "type": "string",
              "value": "zh-CN"
            },
            "confidence": {
              "type": "number",
              "value": 0.96
            }
          }
        }
        """
    )

    payload = backend.generate_json(
        "plan this",
        schema_name="QueryPlan",
    )

    assert payload == {
        "intent": "explanatory",
        "frame_reference": 31,
        "subjects": ["robot_2"],
        "requires_program_trace": True,
        "target_variables": ["robot_2.observed_action"],
        "desired_outcomes": {
            "robot_2.next_action": "LEFT",
        },
        "response_language": "zh-CN",
        "confidence": 0.96,
    }


def test_json_output_contract_precedes_long_task_input() -> None:
    backend, prompts = _backend_with_responses(
        '{"intent":"factual","confidence":0.9}'
    )

    backend.generate_json(
        "large evidence payload",
        schema_name="QueryPlan",
    )

    assert prompts[0].index("OUTPUT CONTRACT") < prompts[0].index(
        "TASK INPUT"
    )
    assert prompts[0].endswith("large evidence payload")


def test_causal_instruct_model_uses_tokenizer_chat_template() -> None:
    class FakeTokenizer:
        chat_template = "configured"

        def apply_chat_template(
            self,
            messages,
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> str:
            assert tokenize is False
            assert add_generation_prompt is True
            return (
                f"SYSTEM={messages[0]['content']}\n"
                f"USER={messages[1]['content']}\nASSISTANT="
            )

    backend = object.__new__(HuggingFaceStructuredTransformer)
    backend.is_encoder_decoder = False
    backend.tokenizer = FakeTokenizer()

    formatted = backend._model_prompt("为什么机器人2向左走？")

    assert "USER=为什么机器人2向左走？" in formatted
    assert formatted.endswith("ASSISTANT=")
