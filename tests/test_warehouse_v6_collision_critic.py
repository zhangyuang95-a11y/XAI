"""Training-only collision-risk labels and group-disjoint validation."""
from __future__ import annotations

import re

import numpy as np
import torch

from backend.training import warehouse_v6_collision_critic as collision
from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor, WarehouseV6Actor
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.layouts import STUDY_MAP_LAYOUT
from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import AGENTS, action_masks


def test_enumerated_labels_are_actual_env_step_events(monkeypatch):
    seen_event_deltas = []
    original_step = WarehouseMultiAgentEnv.step

    def recording_step(self, actions):
        before = self.state.robot_collision_events
        result = original_step(self, actions)
        seen_event_deltas.append(int(self.state.robot_collision_events > before))
        return result

    monkeypatch.setattr(WarehouseMultiAgentEnv, "step", recording_step)
    rows = collision._enumerate_event_rows(
        971000, position_pairs=[((1, 2), (2, 3))])
    np.testing.assert_array_equal(rows.collision_events, seen_event_deltas)
    names = [tuple(ACTIONS[int(action)] for action in pair)
             for pair in rows.joint_actions]
    assert int(rows.collision_events[names.index(("DOWN", "LEFT"))]) == 1
    assert int(rows.collision_events[names.index(("WAIT", "LEFT"))]) == 0
    assert all(rows.masks[index, role, int(action)] > 0.5
               for index, pair in enumerate(rows.joint_actions)
               for role, action in enumerate(pair))


def test_position_pair_partition_never_splits_one_pair_across_actions():
    pairs = np.asarray([
        (first, second)
        for first in STUDY_MAP_LAYOUT.passable_positions
        for second in STUDY_MAP_LAYOUT.passable_positions
        if first != second
    ], dtype=np.int16)
    repeated = np.repeat(pairs, 3, axis=0)
    training, validation = collision._partition_position_pairs(repeated, 971000)
    assert training.any() and validation.any()
    assert not np.any(training & validation)
    assert all(training[index:index + 3].all() or validation[index:index + 3].all()
               for index in range(0, len(repeated), 3))
    train_pairs = {tuple(pair.flat) for pair in repeated[training]}
    valid_pairs = {tuple(pair.flat) for pair in repeated[validation]}
    assert train_pairs.isdisjoint(valid_pairs)
    assert all((second[0], second[1], first[0], first[1]) in train_pairs
               for first, second in pairs if tuple((*first, *second)) in train_pairs)
    assert all((second[0], second[1], first[0], first[1]) in valid_pairs
               for first, second in pairs if tuple((*first, *second)) in valid_pairs)


def test_fitted_critic_receipt_and_masked_risk_are_training_only(monkeypatch):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        critic, report = collision.train_collision_critic(971000)
        _, repeated = collision.train_collision_critic(971000)
    finally:
        torch.set_num_threads(old_threads)
    assert {key: report[key] for key in (
        "data_sha256", "split_sha256", "source_sha256", "model_sha256")
    } == {key: repeated[key] for key in (
        "data_sha256", "split_sha256", "source_sha256", "model_sha256")}
    assert report["validation_passed"] is True
    assert report["position_pair_overlap"] == 0
    assert (report["train_unordered_position_pair_groups"]
            + report["validation_unordered_position_pair_groups"]
            == len(STUDY_MAP_LAYOUT.passable_positions)
            * (len(STUDY_MAP_LAYOUT.passable_positions) - 1) // 2)
    assert report["legal_actions_rewritten_by_environment"] == 0
    assert report["validation"]["positive_rows"] >= 20
    assert report["validation"]["precision"] >= 0.90
    assert report["validation"]["recall"] >= 0.90
    assert report["validation"]["brier"] <= 0.05
    assert report["model_sha256"] == collision.collision_critic_sha256(critic)
    assert re.fullmatch(r"[0-9a-f]{64}", report["data_sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", report["source_sha256"])
    assert report["purpose"] == "offline_BC_gradient_only_not_execution"
    assert all(not parameter.requires_grad for parameter in critic.parameters())

    rows = collision._enumerate_event_rows(
        971000, position_pairs=[((1, 2), (2, 3))])
    observation = torch.from_numpy(rows.observations[:1])
    mask = torch.from_numpy(rows.masks[:1])
    risk = collision.risk_matrix(critic, observation, mask)
    assert risk.shape == (1, len(ACTIONS), len(ACTIONS))
    assert not risk.requires_grad
    assert risk[0, ACTIONS.index("DOWN"), ACTIONS.index("LEFT")] > 0.90
    assert risk[0, ACTIONS.index("WAIT"), ACTIONS.index("LEFT")] < 0.10
    illegal = (mask[0, 0, :, None] <= 0.5) | (mask[0, 1, None, :] <= 0.5)
    assert bool(illegal.any())
    assert torch.count_nonzero(risk[0][illegal]) == 0

    def forbidden_training_risk(*_args, **_kwargs):
        raise AssertionError("training collision critic entered frozen NN execution")

    monkeypatch.setattr(collision, "risk_matrix", forbidden_training_risk)
    frozen = FrozenWarehouseV6Actor.from_model(WarehouseV6Actor())
    env = collision._environment(971001)
    actions, _ = frozen.act(env.state, base_seed=971001)
    legal = action_masks(env.state)
    assert all(legal[index, ACTIONS.index(actions[role])] > 0.5
               for index, role in enumerate(AGENTS))
