"""Regression checks for the independently recorded r4.1 partner-alias diagnosis."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/archive/warehouse_r41_fixed_yield_alias_diagnostic_20260911.json"


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode()).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _core(row: dict) -> dict:
    return {key: value for key, value in row.items()
            if key not in {"partner", "seed"}}


def test_recorded_partner_alias_has_bound_sources_and_eight_exact_groups():
    report = json.loads(REPORT.read_text())
    semantic = dict(report)
    expected = semantic.pop("artifact_semantic_sha256")
    assert _digest(semantic) == expected

    for relative, expected_hash in report["source_receipt"].items():
        assert _file_hash(ROOT / relative) == expected_hash

    groups = report["registered_eight_group_evidence"]
    assert len(groups) == 8
    assert sum(item["paired_scenes"] for item in groups) == 456
    assert all(item["all_core_rows_identical"] for item in groups)
    assert all(item["exact_core_row_matches"] == item["paired_scenes"]
               for item in groups)
    assert all(item["differing_core_rows"] == 0
               and item["differing_fields"] == [] for item in groups)
    assert all(item["skilled_core_rows_sha256"]
               == item["fixed_yield_core_rows_sha256"] for item in groups)

    # Recompute the row-level proof when the local immutable run artifacts are
    # present. The committed JSON remains portable when output/ is not shipped.
    for item in groups:
        path = ROOT / item["episodes_jsonl"]
        if not path.is_file():
            continue
        assert _file_hash(path) == item["episodes_jsonl_sha256"]
        rows = [json.loads(line) for line in path.read_text().splitlines()
                if line.strip()]
        skilled = [_core(row) for row in rows if row["partner"] == "skilled"]
        fixed = [_core(row) for row in rows if row["partner"] == "fixed_yield"]
        assert skilled == fixed
        assert _digest(skilled) == item["skilled_core_rows_sha256"]


def test_root_cause_and_completed_boundary_impact_are_explicit():
    report = json.loads(REPORT.read_text())
    partners_source = (ROOT / "env/warehouse_native/partners.py").read_text()
    original_eval = (ROOT / "backend/training/warehouse_r4_active_evaluation.py").read_text()
    conflict_eval = (ROOT / "backend/training/warehouse_r41_active_evaluation.py").read_text()

    assert 'if kind=="fixed_yield" and agent_id=="robot_2"' in partners_source
    assert 'partner_action(env, "robot_1", partner' in original_eval
    assert 'partner_action(env, "robot_1", partner' in conflict_eval

    impact = report["boundary_impact"]
    assert [item["step"] for item in impact] == [50_000, 100_000, 150_000, 200_000]
    assert all(item["selected_under_recorded_audit"] is False for item in impact)
    assert all(suite["unique_nonduplicated_partner_ai_shutdowns"] > 0
               and suite["cannot_become_selected_by_replacing_only_fixed_yield"]
               for item in impact for suite in item["suites"].values())

    supplement = report["supplemental_200k_corroboration"]
    assert len(supplement) == 2
    assert all(item["all_core_rows_identical"] for item in supplement)
