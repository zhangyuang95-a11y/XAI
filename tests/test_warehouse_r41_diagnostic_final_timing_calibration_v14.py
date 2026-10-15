from __future__ import annotations

import pytest

from backend.training import (
    warehouse_r41_diagnostic_final_timing_calibration_v14 as subject,
)


def _closeout(elapsed_seconds: float) -> dict:
    return {
        "v13_timeout_failure": {
            "controller_materializer_timeout_seconds": 600,
            "public_output_published": False,
            "materializer_output_published": False,
            "retry_allowed": False,
        },
        "burned_candidate_universe": {
            "ranking_input_count": 2160,
            "accepted_scene_count": 64,
            "all_ranking_inputs_retired": True,
            "completed_evaluated_prefix_count": 91,
        },
        "burned_observation_projection": {
            "accepted_scene_count": 64,
            "elapsed_seconds": elapsed_seconds,
            "environment_steps": 12345,
            "row_count": 67890,
            "unique_observation_count": 54321,
        },
        "disposition": {
            "same_v13_final_attempt_retry_permitted": False,
            "v13_candidate_and_program_reusable_without_refit": True,
            "v13_passed_outer_reusable_without_rerun": True,
        },
        "information_boundary": {
            "salt_value_published": False,
            "actions_probabilities_labels_or_program_accessed": False,
            "labeled_rows_generated": False,
        },
    }


@pytest.mark.parametrize(("elapsed", "base", "upper", "timeout"), [
    (550.0, 600.0, 660, 1980),
    (900.0, 900.0, 990, 2970),
])
def test_timing_accepts_any_positive_public_measurement_and_uses_legacy_floor(
    elapsed, base, upper, timeout,
):
    value = subject._timing(_closeout(elapsed))
    assert value["conservative_base_seconds"] == base
    assert value["measured_upper_bound_seconds"] == upper
    assert value["required_timeout_seconds"] == timeout


@pytest.mark.parametrize("elapsed", [0.0, -1.0, float("inf"), float("nan")])
def test_timing_rejects_nonpositive_or_nonfinite_measurements(elapsed):
    with pytest.raises(ValueError, match="timing differs"):
        subject._timing(_closeout(elapsed))

