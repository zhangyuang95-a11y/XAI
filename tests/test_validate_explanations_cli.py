from __future__ import annotations

import pytest

from validate_explanations import _smoke_backend


def test_smoke_backend_uses_question_ir_v2_without_language_call() -> None:
    backend = _smoke_backend("zh")

    payload = backend.generate_json(
        "parse the question once",
        schema_name="QuestionIRV2",
    )

    assert payload["target_entity"] == "robot_2"
    assert payload["intent"] == "explanatory"
    assert payload["primitive_interventions"] == []
    assert "response_language" not in payload
    with pytest.raises(ValueError, match="Unsupported smoke schema"):
        backend.generate_json(
            "language must be detected locally",
            schema_name="LanguageIdentification",
        )


def test_smoke_backend_has_no_online_text_or_claim_repair_path() -> None:
    backend = _smoke_backend("en")

    with pytest.raises(RuntimeError, match="no text generator"):
        backend.generate_text("repair this explanation")
    assert backend.generate_json(
        "realize once",
        schema_name="ExplanationDocumentV2",
    ) == {"sections": []}
