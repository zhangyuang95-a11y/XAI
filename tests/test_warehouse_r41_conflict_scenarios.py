from __future__ import annotations

from copy import deepcopy
import json

import pytest

from backend.training.warehouse_r41_conflict_scenarios import (
    FORBIDDEN_LEGACY_SEEDS,
    build_manifest,
    make_scene,
    select_play_scenes,
    successor_replay_audit,
    transition_closure_audit,
    validate_conflict_manifest,
)
from backend.warehouse_r41_online_runtime import R41ConflictWarehouseEnv
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse_native.r41_conflict import (
    CONFLICT_GRAPH,
    CONFLICT_GRAPH_SHA256,
    CONTRACT,
    CONTRACT_SHA256,
    R41ConflictSamplingError,
    digest,
    task_node_id,
    validate_active_conflict,
)


@pytest.fixture(scope="module")
def manifest(tmp_path_factory):
    source = tmp_path_factory.mktemp("r41-old") / "old.json"
    source.write_text(json.dumps({"splits": {"play": []}}), encoding="utf-8")
    return build_manifest(
        [source],
        counts={
            "train": 2,
            "conflict_validation": 2,
            "final_test": 2,
            "tutorial": 1,
            "play_candidates": 60,
        },
    )


def test_conflict_graph_and_contract_are_strict_and_closed():
    assert CONTRACT["minimum_shared_edge_ratio"] == 0.30
    assert CONTRACT["maximum_shared_edge_ratio"] == 0.55
    assert CONTRACT["all_shortest_route_pairs_must_be_in_band"] is True
    assert "cannot be claimed until a later confirmed step" in CONTRACT["creation_frame_claim_semantics"]
    assert len(CONFLICT_GRAPH["nodes"]) == 20
    assert len(CONFLICT_GRAPH["edges"]) == 14
    assert len(CONFLICT_GRAPH["safe_two_core_nodes"]) == 6
    assert len(CONFLICT_GRAPH["safe_two_core_edges"]) == 6
    assert all(len(CONFLICT_GRAPH["transition_table"][node]) == 2
               for node in CONFLICT_GRAPH["safe_two_core_nodes"])
    closure = transition_closure_audit()
    assert closure["delivery_cases"] == 18
    assert closure["robot_position_cases"] == 6156
    assert closure["underfoot_tier_cases"] > 0


def test_environment_never_calls_ordinary_sampler(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("ordinary sampler called")

    monkeypatch.setattr(WarehouseMultiAgentEnv, "_sample_delivery_job", forbidden)
    env = R41ConflictWarehouseEnv()
    env.reset(seed=4_510_000)
    assert validate_active_conflict(env.state.tasks)["passed"]

    state = deepcopy(env.state)
    task = state.tasks[0]
    state.agents[0].position = task.delivery_position
    state.agents[0].carrying_task_id = task.task_id
    task.status = "carried"
    task.carrier_agent_id = state.agents[0].agent_id
    env.set_state(state)
    env.step({"robot_1": "WAIT", "robot_2": "WAIT"})
    assert validate_active_conflict(env.state.tasks)["passed"]


def test_replacement_is_new_and_prefers_a_clear_pickup():
    env = R41ConflictWarehouseEnv()
    env.reset(seed=4_510_003)
    state = deepcopy(env.state)
    delivered = state.tasks[0]
    state.agents[0].position = delivered.delivery_position
    state.agents[0].carrying_task_id = delivered.task_id
    delivered.status = "carried"
    delivered.carrier_agent_id = state.agents[0].agent_id
    env.set_state(state)
    delivered_node = task_node_id(delivered)
    result = env.step({"robot_1": "WAIT", "robot_2": "WAIT"})
    recreated = max(env.state.tasks, key=lambda task: int(task.task_id.split("_")[1]))
    occupied_positions = {agent.position for agent in env.state.agents}
    assert task_node_id(recreated) != delivered_node
    assert recreated.pickup_position not in occupied_positions
    assert recreated.status == "available"
    assert recreated.carrier_agent_id is None
    creation = result[-1]["r41_conflict"]["created"][0]
    assert creation["new_pickup_on_agent"] is False
    assert creation["spawned_on_agent_endpoint"] == creation["new_delivery_on_agent"]
    assert creation["immediate_task_recreation"] is False


def test_underfoot_strict_successor_is_created_after_pickup_phase():
    env = R41ConflictWarehouseEnv()
    env.reset(seed=4_510_003)
    state = deepcopy(env.state)
    delivered = state.tasks[0]
    remaining = state.tasks[1]
    delivered_node = task_node_id(delivered)
    remaining_node = task_node_id(remaining)
    graph_nodes = {row["node_id"]: row for row in CONFLICT_GRAPH["nodes"]}
    successor_node = next(
        node
        for node in CONFLICT_GRAPH["transition_table"][remaining_node]
        if node != delivered_node
    )
    successor_pickup = tuple(graph_nodes[successor_node]["pickup_position"])
    state.agents[0].position = delivered.delivery_position
    state.agents[0].carrying_task_id = delivered.task_id
    delivered.status = "carried"
    delivered.carrier_agent_id = state.agents[0].agent_id
    state.agents[1].position = successor_pickup
    state.agents[1].carrying_task_id = None
    env.set_state(state)

    result = env.step({"robot_1": "WAIT", "robot_2": "WAIT"})
    created = result[-1]["r41_conflict"]["created"][0]
    task = env.state.task_by_id(created["task_id"])
    assert created["selection_tier"] == "strict_conflict_underfoot_pickup"
    assert created["new_pickup_on_agent"] is True
    assert created["immediate_task_recreation"] is False
    assert task.pickup_position == successor_pickup
    assert task.status == "available"
    assert task.carrier_agent_id is None


def test_snapshot_binds_contract_graph_and_successor_state():
    env = R41ConflictWarehouseEnv()
    env.reset(seed=4_510_010)
    snapshot = env.snapshot()
    assert snapshot["r41_conflict"]["contract_sha256"] == CONTRACT_SHA256
    assert snapshot["r41_conflict"]["conflict_graph_sha256"] == CONFLICT_GRAPH_SHA256
    assert env.branch().snapshot() == snapshot

    for field in ("contract_sha256", "conflict_graph_sha256", "successor_state_sha256"):
        damaged = deepcopy(snapshot)
        damaged["r41_conflict"][field] = "0" * 64
        with pytest.raises(ValueError):
            R41ConflictWarehouseEnv().restore(damaged)


def test_successor_replay_uses_actual_step_and_is_reproducible():
    env = R41ConflictWarehouseEnv()
    env.reset(seed=4_510_020)
    first = successor_replay_audit(env.snapshot())
    second = successor_replay_audit(env.snapshot())
    assert first == second
    assert first["replacement_tasks"] == 16
    assert first["all_replays_equal"] is True
    assert first["all_active_pairs_passed"] is True
    assert first["new_pickup_on_agent"] == 0
    assert first["immediate_task_recreation"] == 0
    assert first["new_endpoint_on_agent"] == first["new_delivery_on_agent"]


def test_legacy_seeds_are_rejected():
    for seed in (FORBIDDEN_LEGACY_SEEDS[0], FORBIDDEN_LEGACY_SEEDS[-1]):
        with pytest.raises(ValueError, match="forbidden"):
            make_scene("bad", "play_candidates", seed)


def test_manifest_has_sixty_candidates_disjoint_splits_and_six_edges(manifest):
    report = validate_conflict_manifest(manifest, replay=True)
    assert report["passed"] is True
    assert report["candidate_count"] == 60
    assert report["split_seed_disjoint"] is True
    assert report["split_successor_state_disjoint"] is True
    assert len(manifest["splits"]["play"]) == 6
    assert len({row["initial_edge_id"] for row in manifest["splits"]["play"]}) == 6
    assert len(manifest["selected_play"]["X"]) == 3
    assert len(manifest["selected_play"]["Y"]) == 3
    assert select_play_scenes(manifest["candidate_pool"])["pairs"] == manifest["selected_play"]["pairs"]


def test_manifest_validation_accepts_real_json_round_trip(manifest):
    persisted = json.loads(json.dumps(manifest, ensure_ascii=False, allow_nan=False))
    report = validate_conflict_manifest(persisted, replay=False)
    assert report["passed"] is True


def test_manifest_validation_fails_closed_on_split_leakage(manifest):
    damaged = deepcopy(manifest)
    damaged["splits"]["validation"] = damaged["splits"].pop("conflict_validation")
    damaged["content_sha256"] = digest({key: value for key, value in damaged.items() if key != "content_sha256"})
    with pytest.raises(ValueError, match="split schema"):
        validate_conflict_manifest(damaged, replay=False)


def test_manifest_validation_fails_closed_on_successor_tampering(manifest):
    damaged = deepcopy(manifest)
    damaged["candidate_pool"][0]["snapshot"]["r41_conflict"]["sampler_draws"] += 1
    damaged["content_sha256"] = digest({key: value for key, value in damaged.items() if key != "content_sha256"})
    with pytest.raises(ValueError):
        validate_conflict_manifest(damaged, replay=False)


def test_explicit_sampling_without_context_is_rejected():
    env = R41ConflictWarehouseEnv()
    with pytest.raises(R41ConflictSamplingError, match="explicit"):
        env._sample_delivery_job(task_index=1, created_frame=0, excluded_positions=set())
