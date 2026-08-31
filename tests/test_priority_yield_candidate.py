from __future__ import annotations

from backend.training.priority_yield_candidate import collect_priority_yield_rows
from env.warehouse.domain import WarehouseConfig
from env.warehouse.layouts import COMPACT_STAGGERED_8X9_LAYOUT
from env.warehouse.navigation import ACTIONS
from env.warehouse.observations import observation_dim
from env.warehouse.policy import MAPPOPolicy


def _legacy_config(**overrides) -> WarehouseConfig:
    return WarehouseConfig(
        rows=COMPACT_STAGGERED_8X9_LAYOUT.rows,
        cols=COMPACT_STAGGERED_8X9_LAYOUT.cols,
        map_layout_id=COMPACT_STAGGERED_8X9_LAYOUT.layout_id,
        **overrides,
    )


def test_priority_yield_rows_are_symmetric_wait_labels_without_teacher_execution() -> None:
    policy = MAPPOPolicy(_legacy_config(horizon=120), device="cpu")
    rows, labels, coverage = collect_priority_yield_rows(
        policy,
        episodes=16,
        seed=92_500,
        only_actor_wait_preferred=False,
    )

    assert rows.shape == (16, observation_dim(policy.environment_config))
    assert set(labels.tolist()) == {ACTIONS.index("WAIT")}
    assert coverage["priority_counts"] == {"robot_1": 8, "robot_2": 8}
    assert coverage["teacher_actions_submitted_to_environment"] == 0
