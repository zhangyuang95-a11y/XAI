from __future__ import annotations

from copy import deepcopy

import pytest

from backend.training import warehouse_r4_conflict_scene_selection as selection
from env.warehouse_native.environment import NativeWarehouseEnv
from env.warehouse_native.scenarios import scenario_fingerprint


def scene(seed: int):
    env = NativeWarehouseEnv()
    env.reset(seed=seed)
    for agent in env.state.agents:
        agent.battery = 100.
    env.state.episode_id = env._episode_counter = 1
    return {"id": f"play_{seed}", "seed": seed, "fingerprint": scenario_fingerprint(env),
            "snapshot": deepcopy(env.snapshot())}


def test_known_high_conflict_geometry_has_robust_overlap_and_both_flow_modes():
    env = NativeWarehouseEnv()
    env.reset(seed=500002)
    result = selection.geometry_metrics(env)
    assert result["passed"]
    assert result["shortest_task_route_shared_edge_ratio_min"] == pytest.approx(.5)
    assert result["shortest_task_route_shared_edge_ratio_max"] == pytest.approx(.5)
    assert result["shared_bridge_edges"] or result["shared_intersections"]
    assert result["same_direction_mission_edges"] > 0
    assert result["opposing_mission_edges"] > 0


def test_seed_scan_excludes_old_play_and_produces_at_least_sixty_distinct_states():
    manifest = {"splits": {"play": [scene(500000)], "train": []}}
    rows, audit = selection.scan_new_seed_states(
        manifest, seed_start=500000, distinct_count=60, maximum_draws=5000)
    assert audit["accepted_distinct_states"] == 60
    assert audit["seed_already_in_source_manifest"] == 1
    assert len({row["seed"] for row in rows}) == 60
    assert len({row["task_signature"] for row in rows}) == 60
    old = NativeWarehouseEnv(); old.restore(manifest["splits"]["play"][0]["snapshot"])
    assert selection._task_signature(old) not in {row["task_signature"] for row in rows}


def test_expanded_scan_can_audit_three_hundred_distinct_observed_seed_states():
    manifest = {"splits": {"play": [scene(500000)], "train": []}}
    rows, audit = selection.scan_new_seed_states(
        manifest, seed_start=510000, distinct_count=300, maximum_draws=5000,
        unique_task_geometries=False)
    assert audit["accepted_distinct_states"] == 300
    assert len({row["observed_state_signature"] for row in rows}) == 300
    assert audit["accepted_unique_task_geometries"] < 300


def test_dynamic_gate_uses_pooled_simple_runs_and_never_ignores_action_overrides():
    rows = []
    for profile in (*selection.BASELINES, "compatible_reference"):
        for continuation_seed in selection.CONTINUATION_SEEDS:
            reference = profile == "compatible_reference"
            rows.append({"profile": profile, "steps": 100, "deliveries": 10 if reference else 8,
                         "conflict_opportunity_frames": 30, "player_risky_action_mass_sum": 15.,
                         "collisions": 10, "recovered_collisions_within_10": 9,
                         "longest_consecutive_collisions": 2, "longest_no_progress_streak": 12,
                         "terminal_consecutive_collisions": 0, "terminal_no_progress_streak": 0,
                         "actor_submission_frames": 100, "actor_action_override_frames": 0,
                         "terminal_reason": "horizon"})
    result = selection.dynamic_metrics(rows)
    assert result["passed"]
    assert result["compatible_reference_mean_deliveries"] == 10
    assert result["best_simple_mean_deliveries"] == 8
    rows[0]["actor_action_override_frames"] = 1
    result = selection.dynamic_metrics(rows)
    assert not result["passed"]
    assert not result["checks"]["actor_actions_submitted_unchanged"]


def test_transient_streak_is_diagnostic_but_terminal_stall_is_persistent_deadlock():
    rows = []
    for profile in (*selection.BASELINES, "compatible_reference"):
        for continuation_seed in selection.CONTINUATION_SEEDS:
            reference = profile == "compatible_reference"
            rows.append({"profile": profile, "steps": 120, "deliveries": 10 if reference else 8,
                         "conflict_opportunity_frames": 36, "player_risky_action_mass_sum": 18.,
                         "collisions": 12, "recovered_collisions_within_10": 11,
                         "longest_consecutive_collisions": 25, "longest_no_progress_streak": 50,
                         "terminal_consecutive_collisions": 0, "terminal_no_progress_streak": 0,
                         "actor_submission_frames": 120, "actor_action_override_frames": 0,
                         "terminal_reason": "horizon"})
    result = selection.dynamic_metrics(rows)
    assert result["passed"]
    assert result["max_consecutive_collisions"] == 25
    assert result["max_no_progress_streak"] == 50
    rows[0]["terminal_no_progress_streak"] = 11
    result = selection.dynamic_metrics(rows)
    assert not result["passed"]
    assert not result["checks"]["no_persistent_terminal_no_progress_deadlock"]


def test_compatible_reference_assigns_a_distinct_public_task_and_avoids_collision():
    found = False
    for seed in range(500000, 500100):
        env = NativeWarehouseEnv()
        env.reset(seed=seed)
        for agent in env.state.agents:
            agent.battery = 100.
        for actor_action in selection.MOVE_DELTAS:
            plan = selection.compatible_reference_plan(env, actor_action)
            if plan["kind"] != "complementary_pickup":
                continue
            assert plan["participant_task_id"] != plan["inferred_actor_task_id"]
            action = selection.compatible_reference_action(env, actor_action)
            assert action in selection._legal_player_actions(env)
            collision = env._resolve_motion(
                env.state, {"robot_1": action, "robot_2": actor_action})[3]
            any_safe = any(not env._resolve_motion(
                env.state, {"robot_1": candidate, "robot_2": actor_action})[3]
                for candidate in selection._legal_player_actions(env))
            assert not any_safe or not collision
            found = True
            break
        if found:
            break
    assert found


def test_collision_recovery_window_is_inclusive_and_never_double_counts():
    pending = [{"frame": 1, "resolved": False}, {"frame": 2, "resolved": False}]
    assert selection.update_collision_recovery(pending, frame=11, progress=False) == 0
    assert selection.update_collision_recovery(pending, frame=11, progress=True) == 2
    assert selection.update_collision_recovery(pending, frame=11, progress=True) == 0
    expired = [{"frame": 1, "resolved": False}]
    assert selection.update_collision_recovery(expired, frame=12, progress=True) == 0


def test_pairing_returns_balanced_xy_and_respects_each_pair():
    rows = []
    for index, (work, conflict, deliveries) in enumerate(
            [(20, .30, 10), (21, .31, 10), (22, .32, 11), (20, .30, 10), (21, .31, 10), (22, .32, 11)]):
        rows.append({"id": f"s{index}", "seed": 600000 + index, "fingerprint": str(index),
                     "task_signature": str(index), "snapshot": {},
                     "geometry": {"initial_joint_work_steps": work,
                                  "shortest_task_route_shared_edge_ratio_min": .4},
                     "dynamic": {"passed": True, "compatible_reference_mean_deliveries": deliveries,
                                 "conflict_opportunity_fraction": conflict, "player_risky_action_mass": .15,
                                 "collision_cancellation_fraction": .13, "reference_absolute_delivery_gap": 2.}})
    result = selection.select_balanced_six(rows)
    assert result is not None
    assert len(result["pairs"]) == 3
    assert {row["id"] for row in (*result["X"], *result["Y"])} == {f"s{i}" for i in range(6)}
    assert result["balance"]["workload_relative_difference"] <= .05
    assert result["balance"]["conflict_relative_difference"] <= .05


def test_deployment_package_has_practice_then_paired_xy_and_unique_physics():
    selected = {"practice": None, "X": [], "Y": [], "pairs": [["x0", "y0"], ["x1", "y1"], ["x2", "y2"]],
                "source_scenario_manifest_sha256": "a" * 64}
    entries = []
    for index in range(7):
        entry = {"id": f"source_{index}", "seed": 700000 + index, "fingerprint": f"fp_{index}",
                 "task_signature": f"task_{index}", "observed_state_signature": f"state_{index}",
                 "snapshot": {"number": index}}
        entries.append(entry)
    selected["practice"] = entries[0]
    selected["X"] = entries[1:4]
    selected["Y"] = entries[4:7]
    package = selection.deployment_scene_package({"configuration": {"horizon": 120}}, selected, "b" * 64)
    assert [row["id"] for row in package["play"]] == [f"play_{i:04d}" for i in range(7)]
    assert [row["selection_source_id"] for row in package["play"]] == [f"source_{i}" for i in range(7)]
    assert package["pairs"] == [[1, 4], [2, 5], [3, 6]]
    duplicate = deepcopy(selected)
    duplicate["Y"][2] = deepcopy(duplicate["Y"][2])
    duplicate["Y"][2]["fingerprint"] = duplicate["X"][0]["fingerprint"]
    with pytest.raises(ValueError, match="distinct seeds and physical fingerprints"):
        selection.deployment_scene_package({"configuration": {}}, duplicate, "b" * 64)


def test_release_builder_selected_scene_contract_is_exact_and_pairs_zip_xy():
    entries = [{"id": f"source_{index}", "seed": 710000 + index,
                "fingerprint": f"fp_{index}", "task_signature": f"task_{index}",
                "observed_state_signature": f"state_{index}", "snapshot": {"number": index}}
               for index in range(7)]
    selected = {"version": selection.VERSION, "actor_sha256": "a" * 64,
                "source_scenario_manifest_sha256": "b" * 64, "practice": entries[0],
                "X": entries[1:4], "Y": entries[4:7],
                "pairs": [["source_1", "source_4"], ["source_2", "source_5"],
                          ["source_3", "source_6"]],
                "balance": {}, "selection_score": 0.0}
    selection.validate_selected_scene_contract(selected)
    invalid = deepcopy(selected)
    invalid["extra"] = True
    with pytest.raises(ValueError, match="unexpected top-level"):
        selection.validate_selected_scene_contract(invalid)
    invalid = deepcopy(selected)
    invalid["pairs"][0].reverse()
    with pytest.raises(ValueError, match=r"zip\(X, Y\)"):
        selection.validate_selected_scene_contract(invalid)
