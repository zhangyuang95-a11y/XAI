from __future__ import annotations

from copy import deepcopy

import numpy as np

from backend.training.warehouse_r4_conflict_scene_selection import CONTINUATION_SEEDS
from backend.training.warehouse_r41_conflict_play_selection import (
    PROTOCOL,
    run_actor_episode,
    select_balanced_six,
)
from backend.training.warehouse_r41_conflict_scenarios import make_scene
from backend.warehouse_r41_online_runtime import R41ConflictWarehouseEnv
from env.warehouse.navigation import ACTIONS


class WaitActor:
    obs_dim = 197

    def __init__(self):
        self.metadata = {"feature_names": list(R41ConflictWarehouseEnv().feature_names)}

    def act(self, observations, deterministic=True):
        probabilities = np.zeros(len(ACTIONS), dtype=np.float32)
        probabilities[ACTIONS.index("WAIT")] = 1.0
        return (
            {agent: "WAIT" for agent in observations},
            {agent: probabilities.copy() for agent in observations},
        )


def _passing_dynamic():
    return {
        "passed": True,
        "conflict_opportunity_fraction": 0.325,
        "player_risky_action_mass": 0.15,
        "collision_cancellation_fraction": 0.13,
        "compatible_reference_mean_deliveries": 5.0,
        "reference_absolute_delivery_gap": 2.0,
    }


def test_full_episode_keeps_actor_submission_and_conflict_successors():
    scene = make_scene("candidate", "play_candidates", 4_510_100)
    row = run_actor_episode(scene, WaitActor(), "fixed_yield", CONTINUATION_SEEDS[0])
    assert row["steps"] == 120
    assert row["actor_submission_frames"] == 120
    assert row["actor_action_override_frames"] == 0
    assert row["active_conflict_pair_checks"] == 120
    assert row["new_pickup_on_agent"] == 0
    assert row["immediate_task_recreation"] == 0
    assert row["new_endpoint_on_agent"] == row["new_delivery_on_agent"]


def test_selection_requires_all_six_distinct_edges_and_exact_balance():
    scenes = []
    seen = set()
    seed = 4_510_200
    while len(scenes) < 6:
        scene = make_scene(f"candidate_{len(scenes)}", "play_candidates", seed)
        seed += 1
        if scene["initial_edge_id"] in seen:
            continue
        seen.add(scene["initial_edge_id"])
        scene["dynamic"] = _passing_dynamic()
        scenes.append(scene)
    selected = select_balanced_six(scenes)
    assert selected is not None
    chosen = [*selected["X"], *selected["Y"]]
    assert len({row["initial_edge_id"] for row in chosen}) == 6
    assert selected["balance"]["workload_relative_difference"] <= 0.05
    assert selected["balance"]["conflict_relative_difference"] <= 0.05

    damaged = deepcopy(scenes)
    for row in damaged:
        row["initial_edge_id"] = scenes[0]["initial_edge_id"]
    assert select_balanced_six(damaged) is None


def test_static_manifest_play_is_explicitly_non_release_placeholder():
    assert PROTOCOL["source_static_play_status"] == "ignored_provisional_not_release_eligible"
    assert PROTOCOL["candidate_count"] == 60
    assert PROTOCOL["runs_per_profile"] == 20
    assert PROTOCOL["full_horizon_replay"] is True
    assert PROTOCOL["additional_pairing_gates"]["six_distinct_initial_conflict_edges"] is True
