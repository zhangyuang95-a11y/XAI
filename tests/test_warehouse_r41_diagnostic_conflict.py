from __future__ import annotations

from copy import deepcopy

import pytest

from backend.training.warehouse_r41_diagnostic_conflict_scenarios import (
    FAMILY_IDS,
    generate_candidate_batch,
)
from backend.warehouse_r41_diagnostic_online_runtime import (
    R41DiagnosticConflictWarehouseEnv,
)
from env.warehouse_native.r41_diagnostic_conflict import (
    CONFLICT_FAMILIES,
    CONFLICT_FAMILIES_SHA256,
    DIAGNOSTIC_CONFLICT_GRAPH,
    DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
    DIAGNOSTIC_CONTRACT_SHA256,
    NODE_BY_ID,
    conflict_family_id,
    digest,
    diagnostic_graph_invariant_audit,
    diagnostic_scene_fingerprint,
    reset_diagnostic_scenario,
)


def test_six_conflict_families_partition_the_robust_edge_closure():
    assert len(CONFLICT_FAMILIES) == 6
    assert {row["family_id"] for row in CONFLICT_FAMILIES} == set(FAMILY_IDS)
    robust = DIAGNOSTIC_CONFLICT_GRAPH["robust_edges"]
    assert len(robust) == 31
    assert len(DIAGNOSTIC_CONFLICT_GRAPH["robust_nodes"]) == 13
    assert {
        edge_id
        for family in CONFLICT_FAMILIES
        for edge_id in family["edge_ids"]
    } == {row["edge_id"] for row in robust}
    assert sum(row["edge_count"] for row in CONFLICT_FAMILIES) == 31
    assert all(row["metrics"]["passed"] for row in robust)


def test_robust_graph_exhaustively_keeps_both_successor_endpoints_clear():
    audit = diagnostic_graph_invariant_audit()
    assert audit["passed"] is True
    assert audit["edge_count"] == 31
    assert audit["single_delivery_contexts"] == 1_116
    assert audit["single_successor_count_min"] >= 1
    assert audit["simultaneous_delivery_contexts"] == 31
    assert audit["simultaneous_first_count_min"] >= 1
    assert audit["simultaneous_second_count_min"] >= 1
    assert audit["both_endpoints_clear"] is True
    assert audit["immediate_recreation_forbidden"] is True
    assert audit["ordinary_sampler_fallback"] is False


def test_small_candidate_batch_is_exactly_balanced_and_restorable():
    scenes, report = generate_candidate_batch(0, per_family=2, maximum_draws=1_000)
    assert len(scenes) == 12
    assert set(report["per_family"]) == set(FAMILY_IDS)
    assert set(report["per_family"].values()) == {2}
    assert len({row["seed"] for row in scenes}) == 12
    assert len({row["fingerprint"] for row in scenes}) == 12
    for scene in scenes:
        assert scene["diagnostic_contract_sha256"] == DIAGNOSTIC_CONTRACT_SHA256
        assert scene["conflict_families_sha256"] == CONFLICT_FAMILIES_SHA256
        env = R41DiagnosticConflictWarehouseEnv()
        reset_diagnostic_scenario(env, scene)
        assert conflict_family_id(env.state.tasks) == scene["family_id"]
        assert diagnostic_scene_fingerprint(env) == scene["fingerprint"]


def test_every_created_successor_keeps_both_endpoints_off_both_robots():
    env = R41DiagnosticConflictWarehouseEnv()
    env.reset(seed=48_100_007)
    for _ in range(12):
        state = env.get_state()
        delivered = state.tasks[0]
        delivered.status = "carried"
        delivered.carrier_agent_id = state.agents[0].agent_id
        delivered.claimed_frame = state.frame
        delivered.claimed_battery = 100.0
        state.agents[0].position = delivered.delivery_position
        state.agents[0].carrying_task_id = delivered.task_id
        state.agents[0].battery = 100.0
        occupied = {
            tuple(delivered.delivery_position),
            *(tuple(task.pickup_position) for task in state.tasks),
        }
        state.agents[1].position = next(
            position for position in sorted(env.layout.passable_positions)
            if position not in occupied
        )
        state.agents[1].carrying_task_id = None
        state.agents[1].battery = 100.0
        env.set_state(state)
        info = env.step({"robot_1": "WAIT", "robot_2": "WAIT"})[-1]
        robot_positions = {tuple(agent.position) for agent in env.state.agents}
        for row in info["r41_diagnostic_conflict"]["created"]:
            node = NODE_BY_ID[row["node_id"]]
            assert not {
                tuple(node["pickup_position"]), tuple(node["delivery_position"])
            } & robot_positions
            assert row["new_endpoint_on_agent"] is False
            assert row["new_pickup_on_agent"] is False
            assert row["new_delivery_on_agent"] is False
            assert row["immediate_task_recreation"] is False


def test_diagnostic_snapshot_tampering_is_rejected():
    env = R41DiagnosticConflictWarehouseEnv()
    env.reset(seed=46_100_007)
    snapshot = env.snapshot()
    for field in (
        "contract_sha256",
        "conflict_families_sha256",
        "binding_sha256",
    ):
        damaged = deepcopy(snapshot)
        damaged["r41_diagnostic_conflict"][field] = "0" * 64
        with pytest.raises(ValueError):
            R41DiagnosticConflictWarehouseEnv().restore(damaged)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("new_pickup_on_agent", True),
        ("new_delivery_on_agent", True),
        ("new_endpoint_on_agent", True),
        ("spawned_on_agent_endpoint", True),
        ("occupied_spawn_endpoints", [[1, 1]]),
        ("immediate_task_recreation", True),
        ("selection_tier", "legacy_fallback"),
    ),
)
def test_self_consistent_snapshot_cannot_hide_invalid_creation_history(field, value):
    env = R41DiagnosticConflictWarehouseEnv()
    env.reset(seed=48_100_007)
    damaged = deepcopy(env.snapshot())
    conflict = damaged["r41_conflict"]
    conflict["creation_history"][0][field] = value
    conflict["successor_state_sha256"] = digest(
        {
            "rng": damaged["rng"],
            "sampler_draws": conflict["sampler_draws"],
            "active_task_nodes": conflict["active_task_nodes"],
            "creation_history": conflict["creation_history"],
        }
    )
    diagnostic = damaged["r41_diagnostic_conflict"]
    diagnostic["base_successor_state_sha256"] = conflict["successor_state_sha256"]
    diagnostic["binding_sha256"] = digest(
        {
            "diagnostic_contract_sha256": DIAGNOSTIC_CONTRACT_SHA256,
            "diagnostic_conflict_graph_sha256": DIAGNOSTIC_CONFLICT_GRAPH_SHA256,
            "conflict_families_sha256": CONFLICT_FAMILIES_SHA256,
            "base_successor_state_sha256": conflict["successor_state_sha256"],
        }
    )
    with pytest.raises(ValueError, match="Diagnostic creation history differs"):
        R41DiagnosticConflictWarehouseEnv().restore(damaged)
