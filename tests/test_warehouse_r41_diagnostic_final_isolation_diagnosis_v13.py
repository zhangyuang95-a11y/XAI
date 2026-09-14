from __future__ import annotations

import inspect

from backend.training import warehouse_r41_diagnostic_final_isolation_diagnosis_v13 as subject


def test_diagnosis_freezes_the_two_mismatched_replay_contracts() -> None:
    value = subject._code_contract()
    assert value["selection_projection_partner_rng"] == (
        "17000 + partner_index * 1000 + scene_index")
    assert value["audit_collector_partner_rng"] == (
        "41_900_000 + scene_index * 101 + partner_index")
    assert value["selection_projection_critical_anchor_period"] == 10
    assert value["audit_collector_critical_anchor_period"] == 5
    assert value["projection_contracts_match"] is False


def test_diagnosis_cannot_invoke_protected_operations() -> None:
    source = inspect.getsource(subject)
    assert "_run_materializer" not in source
    assert "run_final_once" not in source
    assert "_read_committed_salt" not in source
    assert subject.contract()["protected_salt_access"] is False
    assert subject.contract()["new_outer_or_final_attempt"] is False
