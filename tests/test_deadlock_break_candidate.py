from __future__ import annotations

from backend.training.deadlock_break_candidate import (
    _configure_structural_case,
    collect_deadlock_break_rows,
)
from env.warehouse.domain import collaborative_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.navigation import ACTIONS
from env.warehouse.observations import observation_dim
from env.warehouse.policy import MAPPOPolicy
import numpy as np
import torch


def test_deadlock_break_rows_only_label_a_safe_non_wait_mover() -> None:
    policy = MAPPOPolicy(collaborative_study_config(horizon=120), device="cpu")
    rows, labels, coverage = collect_deadlock_break_rows(
        policy,
        cases=96,
        seed=93_000,
        require_actor_joint_wait=False,
    )

    assert rows.shape[1] == observation_dim(policy.environment_config)
    assert len(rows) == len(labels)
    assert "WAIT" in {ACTIONS[int(label)] for label in labels}
    assert any(ACTIONS[int(label)] != "WAIT" for label in labels)
    assert len(rows) == 2 * coverage["teacher_single_move_states"]
    assert coverage["teacher_actions_submitted_to_environment"] == 0
    assert set(coverage["family_counts"]) == {"horizontal_single_lane"}
    assert coverage["horizontal_nondual_only"] is True


def test_deadlock_break_structure_is_gated_by_observed_wait_history() -> None:
    environment = WarehouseMultiAgentEnv(collaborative_study_config(horizon=120))
    environment.reset(seed=93_100)

    _configure_structural_case(environment, case=0)

    assert 3 <= environment.get_state().ineffective_joint_wait_streak <= 7


def test_escape_residual_is_exactly_zero_outside_its_frozen_state_gate() -> None:
    policy = MAPPOPolicy(collaborative_study_config(horizon=120), device="cpu")
    environment = WarehouseMultiAgentEnv(policy.environment_config)
    environment.reset(seed=93_200)
    _configure_structural_case(environment, case=9)

    def logits() -> np.ndarray:
        observations = environment.observations()
        rows = np.stack([observations[key] for key in sorted(observations)])
        with torch.no_grad():
            return policy.network.actor_logits(torch.from_numpy(rows)).numpy()

    baseline = logits()
    with torch.no_grad():
        policy.network.deadlock_escape_action_head[2].bias[
            ACTIONS.index("UP")
        ] = 1.0
    assert not np.allclose(logits(), baseline)

    state = environment.get_state()
    state.ineffective_joint_wait_streak = 0
    environment.set_state(state)
    gated_baseline = logits()
    with torch.no_grad():
        policy.network.deadlock_escape_action_head[2].bias.zero_()
    np.testing.assert_allclose(gated_baseline, logits(), atol=0.0, rtol=0.0)
