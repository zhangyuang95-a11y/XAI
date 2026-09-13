from pathlib import Path

from backend.training import warehouse_r41_diagnostic_rcpd_v7 as subject


def _robustness(value: float):
    return {
        "folds": [
            {
                "fold": fold,
                "scene_count": 16,
                "direction_fidelity": value,
                "direction_by_group": {
                    group: {"fidelity": value, "scenes": 12, "pairs": 20}
                    for group in subject.GROUPS
                },
            }
            for fold in range(subject.SCENE_FOLD_COUNT)
        ]
    }


def _metrics(value: float = 0.95):
    return {
        "overall": {"fidelity": value, "rows": 100, "scenes": 64},
        "nonwait": {"fidelity": value, "rows": 80, "scenes": 64},
        "critical": {
            group: {"fidelity": value, "rows": 30, "scenes": 64}
            for group in subject.GROUPS
        },
        "effective_intervention_direction": {
            "fidelity": value, "pairs": 30, "scenes": 64,
            "by_group": {
                group: {"fidelity": value, "pairs": 20, "scenes": 32}
                for group in subject.GROUPS
            },
        },
        "mean_kl": 0.1,
    }


def test_v7_uses_public_inputs_and_a_fresh_development_supplement():
    contract = subject.contract()
    assert subject.VERSION == "warehouse-r41-diagnostic-rcpd.v7"
    assert subject.MEMBER_COUNT == 7
    assert contract["fit_access"]["prediction_input"] == "public observation vector only"
    assert contract["fit_access"]["actor_logits_as_program_input"] is False
    assert contract["fit_access"]["actor_hidden_states"] is False
    assert contract["fit_access"]["final_test"] is False
    assert contract["development_supplement_version"].endswith(".v1")
    assert "ui/warehouse_alignment_online_server.py" not in subject.producer_sources()


def test_active_v7_chain_requires_cycle_free_designation_and_frozen_manifest():
    assert subject.designation_binding.EXPECTED_DESIGNATION_SHA256 == (
        "b42323e3bc4543c4f4e1af96be4de4d90489a38459240bfb494dcc2d6120a815"
    )
    sources = subject.producer_sources()
    assert "backend/training/warehouse_r41_diagnostic_designation_v2.py" in sources
    assert "backend/training/warehouse_r41_diagnostic_frozen_manifest_v2.py" in sources


def test_legacy_v7_has_no_raw_final_identity_reader():
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert "protected_final_identities" not in source


def test_every_scene_fold_must_pass_direction_gate():
    assert subject._candidate_gate(_metrics(), _robustness(0.85))["passed"] is True
    weak = _robustness(0.85)
    weak["folds"][2]["direction_by_group"]["shared_pickup"]["fidelity"] = 0.849
    result = subject._candidate_gate(_metrics(), weak)
    assert result["passed"] is False
    assert result["checks"]["fold_2_direction_shared_pickup"] is False
