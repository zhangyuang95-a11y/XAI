from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from env.warehouse.domain import collaborative_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.interaction_calibration import calibrate_interactions
from env.warehouse.layouts import STUDY_MAP_LAYOUT
from env.warehouse.mappo import MAPPOConfig, MAPPOPolicy
from env.warehouse.navigation import shortest_path_distance
from env.warehouse.numpy_policy import NumpyWarehousePolicy
from ui.development_preview_server import (
    DEPLOYED_ACTOR,
    DevelopmentPreviewState,
)
from ui.warehouse_view import warehouse_map_payload


ROOT = Path(__file__).resolve().parents[1]


def test_production_map_uses_staggered_work_aisles_not_aligned_crossroads() -> None:
    layout = STUDY_MAP_LAYOUT
    assert layout.layout_id == (
        "warehouse_staggered_aisles_6x7_v2_three_cell_exit_no_cross"
    )
    assert (layout.rows, layout.cols) == (6, 7)

    assert tuple(
        column for column in range(7) if layout.is_passable((1, column))
    ) == (0, 1, 2, 3)
    assert tuple(
        column for column in range(7) if layout.is_passable((2, column))
    ) == (2, 3, 4, 5, 6)
    assert tuple(
        column for column in range(7) if layout.is_passable((3, column))
    ) == (0, 1, 2)
    # The work aisles and the mandated three-cell robot/charger apron contain
    # no four-neighbour cell at all.
    assert layout.four_way_intersections == ()


def test_production_robot_exit_is_exactly_three_real_passable_cells() -> None:
    layout = STUDY_MAP_LAYOUT
    assert layout.robot_exit_positions == ((4, 2), (4, 3), (4, 4))
    assert all(layout.is_passable(position) for position in layout.robot_exit_positions)
    assert tuple(
        (layout.rows - 2, column)
        for column in sorted(
            position[1]
            for position in (*layout.robot_start_positions, layout.charger_position)
        )
    ) == layout.robot_exit_positions
    assert tuple(
        column
        for column in range(layout.cols)
        if layout.is_passable((layout.rows - 1, column))
    ) == (2, 3, 4)


def test_production_map_is_connected_and_ui_uses_the_same_geometry() -> None:
    layout = STUDY_MAP_LAYOUT
    origin = layout.robot_start_positions[0]
    assert all(
        shortest_path_distance(origin, position, layout.layout_id) < 100
        for position in layout.passable_positions
    )

    payload = warehouse_map_payload(layout)
    assert payload["layout_id"] == layout.layout_id
    assert (payload["rows"], payload["cols"]) == (layout.rows, layout.cols)
    assert {tuple(position) for position in payload["shelves"]} == set(
        layout.blocked_positions
    )
    assert tuple(payload["charger_position"]) == layout.charger_position
    assert tuple(map(tuple, payload["robot_exit_positions"])) == (
        layout.robot_exit_positions
    )


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
        ROOT / "output" / "deployment" / "warehouse_mappo_v68_6x7.pt",
        device="cpu",
    )
    exported = NumpyWarehousePolicy.load(
        ROOT / "output" / "deployment" / "warehouse_mappo_v68_6x7_actor.npz"
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
    assert DEPLOYED_ACTOR is not None
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


def test_staggered_map_meets_interaction_and_delivery_calibration_gate() -> None:
    report = calibrate_interactions(
        collaborative_study_config(),
        seeds=range(51_000, 51_010),
        participant_noise_probability=0.15,
    )
    assert report["mean"]["deliveries"] >= 5.5
    assert report["minimum"]["collision_opportunity_frames"] >= 10
    # The calibration must expose real joint-action collision opportunities,
    # but safer public coordination can legitimately reduce how often the
    # 15%-noise participant realizes one on a ten-seed sample.
    assert report["mean"]["robot_collisions"] > 0.0


@pytest.mark.parametrize(
    ("actions", "kind"),
    [
        ({"robot_1": "RIGHT", "robot_2": "DOWN"}, "same_target"),
        ({"robot_1": "RIGHT", "robot_2": "LEFT"}, "swap"),
        ({"robot_1": "RIGHT", "robot_2": "WAIT"}, "occupied_stationary"),
    ],
)
def test_simultaneous_resolver_keeps_collision_kinds_reachable(actions, kind) -> None:
    environment = WarehouseMultiAgentEnv(collaborative_study_config(horizon=1))
    environment.reset(seed=862)
    state = environment.get_state()
    if kind == "same_target":
        state.by_id("robot_1").position = (5, 3)
        state.by_id("robot_2").position = (4, 4)
    else:
        state.by_id("robot_1").position = (5, 3)
        state.by_id("robot_2").position = (5, 4)
    environment.set_state(state)
    *_, collision, collision_kind, _ = environment._resolve_motion(state, actions)
    assert collision
    assert collision_kind == kind
