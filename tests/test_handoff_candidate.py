from __future__ import annotations

from backend.training.handoff_candidate import collect_handoff_rows
from env.warehouse.domain import WarehouseConfig
from env.warehouse.navigation import ACTIONS
from env.warehouse.observations import observation_dim
from env.warehouse.policy import MAPPOPolicy


def test_handoff_refit_rows_cover_both_priorities_without_executing_teacher() -> None:
    policy = MAPPOPolicy(WarehouseConfig(horizon=120), device="cpu")
    rows, labels, categories, coverage = collect_handoff_rows(
        policy,
        episodes=16,
        seed=91_500,
    )
    assert rows.shape == (32, observation_dim(policy.environment_config))
    assert labels.shape == (32,)
    assert set(categories) == {"charger_queue"}
    assert coverage["priority_counts"] == {"occupant": 8, "waiter": 8}
    assert coverage["teacher_actions_submitted_to_environment"] == 0
    assert coverage["label_counts"]["WAIT"] >= 16
    assert all(0 <= int(label) < len(ACTIONS) for label in labels)
