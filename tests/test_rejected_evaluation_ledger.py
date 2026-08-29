from __future__ import annotations

from pathlib import Path

import pytest

from backend.training.register_rejected_evaluation import (
    register_rejected_evaluation,
)
from backend.training.seed_ledger import reserve_evaluation_seed_span


def _report(*, passing: bool = False):
    return {
        "formal_candidate": passing,
        "seed_ranges": {
            "ai_ai": [12_000_004, 1_000],
            "multi_partner": {
                "coordinated": [12_020_004, 250],
                "hesitant": [12_080_004, 250],
            },
            "occupied_charger_handoff": [12_190_004, 200],
        },
        "acceptance_checks": {
            "collision": True,
            "loaded_detour": passing,
        },
        "artifact_hashes": {"model": "abc"},
    }


def test_rejected_formal_span_is_conservatively_reserved_and_idempotent() -> None:
    summary = {
        "seed_ledger": {
            "schema": "warehouse-training-seed-ledger.v1",
            "reserved_intervals": [],
        }
    }
    once = register_rejected_evaluation(
        summary,
        _report(),
        report_path=Path("rejected.json"),
        report_sha256="def",
    )
    twice = register_rejected_evaluation(
        once,
        _report(),
        report_path=Path("rejected.json"),
        report_sha256="def",
    )
    assert twice == once
    assert once["seed_ledger"]["reserved_intervals"] == [
        {
            "name": "rejected_formal_evaluation_12000004",
            "start": 12_000_004,
            "end_exclusive": 12_190_204,
        }
    ]
    assert once["rejected_formal_evaluations"][0]["failed_checks"] == [
        "loaded_detour"
    ]


def test_passing_report_cannot_be_registered_as_rejected() -> None:
    with pytest.raises(ValueError, match="passing formal report"):
        register_rejected_evaluation(
            {"seed_ledger": {"reserved_intervals": []}},
            _report(passing=True),
            report_path=Path("passing.json"),
            report_sha256="def",
        )


def test_nested_development_ranges_are_added_to_candidate_ledger() -> None:
    updated = reserve_evaluation_seed_span(
        {"schema": "warehouse-training-seed-ledger.v1", "reserved_intervals": []},
        _report()["seed_ranges"],
        name="final_candidate_development_evaluation",
    )
    assert updated["reserved_intervals"] == [
        {
            "name": "final_candidate_development_evaluation",
            "start": 12_000_004,
            "end_exclusive": 12_190_204,
        }
    ]
