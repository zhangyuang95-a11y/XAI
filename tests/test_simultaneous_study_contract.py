from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from env.warehouse.domain import collaborative_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.interaction_calibration import calibrate_interactions
from env.warehouse.mappo import MAPPOConfig, MAPPOPolicy
from env.warehouse.numpy_policy import NumpyWarehousePolicy
from ui.development_preview_server import (
    DEPLOYED_ACTOR,
    DevelopmentPreviewState,
)


ROOT = Path(__file__).resolve().parents[1]


def test_independent_actor_distribution_cannot_see_participant_current_action() -> None:
    config = collaborative_study_config(horizon=8)
    environment = WarehouseMultiAgentEnv(config)
    observations, _ = environment.reset(seed=861)
    policy = MAPPOPolicy(config, MAPPOConfig(hidden_dim=16, seed=861))

    _, first = policy.act(observations, environment.global_state(), deterministic=True)
    _, second = policy.act(
        deepcopy(observations), environment.global_state(), deterministic=True
    )

    # These participant commands are applied downstream after policy.act.
    participant_up = "UP"
    participant_wait = "WAIT"
    assert participant_up != participant_wait
    assert first["robot_2"].probabilities == second["robot_2"].probabilities
    assert first["robot_2"].logits == second["robot_2"].logits


def test_development_preview_ai_action_is_independent_of_current_human_command() -> None:
    first = DevelopmentPreviewState()
    second = DevelopmentPreviewState()
    first.command(
        {"operation_id": "start-a", "command": "start", "payload": {"locale": "en"}}
    )
    second.command(
        {"operation_id": "start-b", "command": "start", "payload": {"locale": "en"}}
    )
    first.command(
        {"operation_id": "task-a", "command": "begin_task1", "payload": {}}
    )
    second.command(
        {"operation_id": "task-b", "command": "begin_task1", "payload": {}}
    )
    before = first.environment.get_state()
    second.environment.set_state(before)

    first._advance_round("LEFT")
    second._advance_round("WAIT")

    assert first.round_frame.actions["robot_2"] == second.round_frame.actions["robot_2"]


def test_render_actor_is_the_exported_neural_checkpoint() -> None:
    checkpoint = MAPPOPolicy.load(
        ROOT / "output" / "deployment" / "warehouse_mappo_v37.pt",
        device="cpu",
    )
    exported = NumpyWarehousePolicy.load(
        ROOT / "output" / "deployment" / "warehouse_mappo_v37_actor.npz"
    )
    environment = WarehouseMultiAgentEnv(collaborative_study_config())
    observations, _ = environment.reset(seed=40_221)
    for agent_id in environment.agent_ids:
        tensor = torch.as_tensor(
            observations[agent_id][None, :], dtype=torch.float32
        )
        with torch.no_grad():
            expected = checkpoint.network.actor_logits(tensor)[0].cpu().numpy()
        actual = exported.logits(observations[agent_id])
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)
    assert exported.artifact_sha256 == DEPLOYED_ACTOR.artifact_sha256
    assert exported.metadata.checkpoint_sha256


def test_public_preview_reports_and_uses_formal_neural_actor() -> None:
    state = DevelopmentPreviewState()
    view = state.view()
    assert view["study"]["formal_policy_loaded"] is True
    assert view["study"]["policy_model_version"] == DEPLOYED_ACTOR.metadata.model_version
    assert view["study"]["policy_artifact_sha256"] == DEPLOYED_ACTOR.artifact_sha256
    frame = state.tutorial_frames[1]
    assert set(frame.action_distributions) == {"robot_1", "robot_2"}
    assert all(
        len(distribution.probabilities) == 5
        for distribution in frame.action_distributions.values()
    )


def test_public_transition_animates_both_results_from_one_joint_step() -> None:
    state = DevelopmentPreviewState()
    state.command(
        {"operation_id": "start", "command": "start", "payload": {"locale": "en"}}
    )
    state.command(
        {"operation_id": "task", "command": "begin_task1", "payload": {}}
    )
    state._advance_round("LEFT")
    transition = state.round_frame.transition
    assert transition is not None
    assert transition["from_frame"] == 0
    assert transition["to_frame"] == 1
    assert {item["id"] for item in transition["agents"]} == {
        "robot_1",
        "robot_2",
    }


def test_compact_map_meets_interaction_and_delivery_calibration_gate() -> None:
    report = calibrate_interactions(
        collaborative_study_config(),
        seeds=range(51_000, 51_010),
        participant_noise_probability=0.15,
    )
    assert report["mean"]["deliveries"] >= 6.0
    assert report["minimum"]["collision_opportunity_frames"] >= 20
    assert report["mean"]["robot_collisions"] >= 0.5


@pytest.mark.parametrize(
    ("actions", "kind"),
    [
        ({"robot_1": "RIGHT", "robot_2": "LEFT"}, "same_target"),
        ({"robot_1": "RIGHT", "robot_2": "LEFT"}, "swap"),
        ({"robot_1": "RIGHT", "robot_2": "WAIT"}, "occupied_stationary"),
    ],
)
def test_simultaneous_resolver_keeps_collision_kinds_reachable(actions, kind) -> None:
    environment = WarehouseMultiAgentEnv(collaborative_study_config(horizon=1))
    environment.reset(seed=862)
    state = environment.get_state()
    if kind == "same_target":
        state.by_id("robot_1").position = (3, 3)
        state.by_id("robot_2").position = (3, 5)
    else:
        state.by_id("robot_1").position = (3, 3)
        state.by_id("robot_2").position = (3, 4)
    environment.set_state(state)
    *_, collision, collision_kind, _ = environment._resolve_motion(state, actions)
    assert collision
    assert collision_kind == kind
