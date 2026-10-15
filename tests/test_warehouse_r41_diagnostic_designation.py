from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from backend.training import warehouse_r41_diagnostic_designation as designation


ZERO = "0" * 64


def _components(root: Path):
    result = {}
    for name in designation.ARTIFACT_NAMES:
        suffix = ".npz" if name == "actor" else ".json"
        path = root / (name + suffix)
        path.write_bytes((name + "\n").encode())
        result[name] = path
    return result


def _checked(root: Path, components):
    bindings = {name: ZERO for name in designation.BINDING_FIELDS}
    return {
        "bindings": bindings,
        "artifacts": {
            name: {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in components.items()
        },
        "evidence": {
            "training_budget_exhausted": True,
            "all_frozen_checkpoints_failed_behavior_gate": True,
            "terminal_actor_selected_under_frozen_gate": False,
            "terminal_dual_evaluation_status": "failed",
            "action_authority_exact": True,
            "action_override_count": 0,
            "historical_closeout_unchanged": True,
            "designation_authority": "explicit_user_instruction_after_failure_closeout",
        },
        "sources": {"source.py": ZERO},
    }


def test_designation_waives_only_behavior_and_never_claims_formal_eligibility(
        monkeypatch, tmp_path):
    monkeypatch.setattr(designation, "ROOT", tmp_path)
    components = _components(tmp_path)
    checked = _checked(tmp_path, components)
    calls = []
    monkeypatch.setattr(
        designation, "validate_components",
        lambda value: calls.append(dict(value)) or deepcopy(checked),
    )
    output = tmp_path / "diagnostic_actor_designation.json"
    built = designation.build_designation(components, output=output)
    saved = designation.read_saved_designation(
        output, expected_sha256=sha256(output.read_bytes()).hexdigest(),
        components=components,
    )
    assert built == saved
    assert saved["status"] == "designated_terminal_actor_for_internal_diagnostic"
    assert saved["behavior_performance_gate_passed"] is False
    assert saved["behavior_performance_gate_waived"] is True
    assert saved["waiver_scope"] == ["behavior_performance"]
    assert saved["runtime_action_override"] is False
    assert saved["formal_ready"] is False
    assert saved["formal_sample_eligible"] is False
    assert saved["human_explanation_effect_validated"] is False
    assert len(calls) == 3


@pytest.mark.parametrize(("field", "value"), [
    ("formal_ready", True),
    ("formal_sample_eligible", True),
    ("runtime_action_override", True),
    ("waiver_scope", ["behavior_performance", "explanation"]),
])
def test_saved_designation_rejects_scope_or_authority_expansion(
        monkeypatch, tmp_path, field, value):
    monkeypatch.setattr(designation, "ROOT", tmp_path)
    components = _components(tmp_path)
    checked = _checked(tmp_path, components)
    monkeypatch.setattr(designation, "validate_components", lambda _: deepcopy(checked))
    output = tmp_path / "designation.json"
    designation.build_designation(components, output=output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload[field] = value
    output.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(ValueError, match="Exact non-formal"):
        designation.read_saved_designation(
            output, expected_sha256=sha256(output.read_bytes()).hexdigest(),
            components=components,
        )


def test_designation_constants_bind_the_actual_terminal_evidence():
    assert designation.DESIGNATED_STEP == 2_000_000
    assert designation.EXPECTED_ACTOR_SHA256.startswith("4ac2ba77")
    assert designation.EXPECTED_ACTOR_PARAMETERS_SHA256.startswith("fc9095d0")
    assert designation.EXPECTED_PROTOCOL_FILE_SHA256.startswith("374ae398")
    assert designation.EXPECTED_DUAL_EVALUATION_SHA256.startswith("cb095a65")
    assert designation.EXPECTED_LEDGER_SHA256.startswith("7ad59b57")
    assert designation.EXPECTED_CLOSEOUT_SHA256.startswith("a29539e6")
    assert set(designation.ARTIFACT_NAMES) == {
        "actor", "protocol", "training_ledger", "dual_evaluation",
        "failure_closeout",
    }


def test_designation_source_closure_contains_new_boundary():
    sources = designation.source_closure()
    for name in (
        "backend/training/warehouse_r41_diagnostic_designation.py",
        "scripts/build_warehouse_r41_diagnostic_designation.py",
        "backend/training/warehouse_r41_failure_closeout.py",
        "backend/training/warehouse_r41_training_ledger.py",
        "env/warehouse_native/policy.py",
    ):
        assert name in sources
        assert len(sources[name]) == 64
