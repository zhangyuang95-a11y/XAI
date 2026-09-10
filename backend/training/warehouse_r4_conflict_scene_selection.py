"""Pre-registered geometry and Actor-conditioned audit for r4 play scenes.

This development tool selects *initial states*, never a neural checkpoint.  It
does not read participant records.  Simple participant conventions are kept
independent of the Actor action; a separately named compatible reference may
read the frozen Actor's current proposal in order to avoid an immediate
physical conflict.  The deployed Actor is never modified or overridden.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from itertools import permutations
import json
import math
from pathlib import Path
import random
from statistics import mean
import time
from typing import Any, Mapping, Sequence

from backend.training.warehouse_native_simple_baselines import baseline_action
from backend.warehouse_alignment_online_runtime import (
    OnlineAlignmentRuntime, OnlinePublicFeedbackEnvironment,
)
from env.warehouse.layouts import get_map_layout
from env.warehouse.navigation import ACTIONS, MOVE_DELTAS, shortest_path_distance
from env.warehouse_native.environment import NativeWarehouseEnv
from env.warehouse_native.partners import partner_action
from env.warehouse_native.policy import NumPyNativeActor
from env.warehouse_native.scenarios import scenario_fingerprint


VERSION = "warehouse-r4-high-conflict-play-selection.v3"
ROOT = Path(__file__).resolve().parents[2]
BASELINES = ("right_of_way", "region_spillover", "task_spillover", "leader_follower", "fixed_yield")
CONTINUATION_SEEDS = tuple(range(774_100, 774_120))
GEOMETRY = {
    "minimum_shared_edge_ratio": .25,
    "maximum_shared_edge_ratio": .55,
    "require_all_shortest_route_pairs_within_band": True,
    "require_shared_bridge_or_intersection": True,
    "require_opposing_and_same_direction_mission_edges": True,
}
DYNAMIC = {
    "conflict_opportunity_fraction": [.25, .40],
    "player_risky_action_mass": [.10, .20],
    "collision_cancellation_fraction": [.08, .18],
    "reference_delivery_gap_absolute": 2.,
    "reference_delivery_gap_relative": .15,
    "collision_recovery_within_steps": 10,
    "minimum_collision_recovery_rate": .90,
    "reject_zero_delivery_collisions_above": 50,
    "persistent_deadlock_terminal_reason": "horizon",
}
PAIRING = {
    "pair_reference_delivery_difference_max": 1.,
    "pair_conflict_opportunity_difference_max": .03,
    "group_workload_relative_difference_max": .05,
    "group_conflict_relative_difference_max": .05,
    "search_order": "all passing scenes ordered by fixed gate-centering score; first fully feasible six-scene solution",
}
ONLINE_REWARD = {"version": "warehouse-native-score-pbrs.v2", "native_score_scale": .01,
                 "static_wall_command_cost": .02, "potential_scale": .25, "gamma": .99,
                 "shared_reward": True, "pickup_bonus": 0., "charge_bonus": 0.}
_WORKER_ACTOR: NumPyNativeActor | None = None
PROTOCOL = {
    "version": VERSION,
    "scope": "development scene selection; never Actor selection or human-effect estimation",
    "minimum_distinct_new_seed_states": 60,
    "default_distinct_seed_states": 120,
    "old_play_scenes_excluded": True,
    "all_existing_manifest_seed_values_excluded": True,
    "old_play_task_geometries_excluded": True,
    "initial_battery": [100., 100.],
    "geometry": GEOMETRY,
    "dynamic": DYNAMIC,
    "simple_baselines": list(BASELINES),
    "runs_per_baseline": len(CONTINUATION_SEEDS),
    "continuation_rng_seeds": list(CONTINUATION_SEEDS),
    "continuation_definition": "same initial state and headings; only future task-sampler RNG differs",
    "metric_definitions": {
        "conflict_opportunity": "pre-action frame where at least one static-legal robot_1 command would physically collide with the Actor's exact robot_2 command",
        "player_risky_action_mass": "uniform mass of those collision-causing commands among robot_1's static-legal commands, averaged over simple-baseline frames",
        "collision_cancellation": "the submitted simple-baseline/Actor pair produces a robot collision and physics cancels movement",
        "task_progress": "pickup, delivery, or strict increase in the environment's public remaining-work potential",
        "recovery": "task progress occurs during one of the ten confirmed steps after a collision frame",
        "persistent_deadlock": "the episode reaches the horizon with more than ten consecutive terminal frames of collisions or no public task progress; a transient streak that later recovers is diagnostic only",
    },
    "compatible_reference": "public actor-proposal complement: infer the Actor-side available task from its submitted direction, break shared-route ambiguity by minimum joint pickup work, then take a collision-free shortest step toward the complementary public task or charger",
    "compatible_reference_revision": "v1 merely ran the assertive nearest-task convention until an immediate collision; the r4 active-Actor diagnostic showed that it duplicated the Actor and underperformed fixed_yield in all 36 geometry-pass scenes, so it did not establish coordinated recoverability",
    "actor_role": "robot_2 deterministic raw argmax; no post-policy override or action re-selection",
    "pairing": PAIRING,
    "participant_data_read": False,
    "final_test_rollouts": 0,
}


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return sha256(canonical(value).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _neighbors(layout, position: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple((position[0] + delta[0], position[1] + delta[1])
                 for delta in MOVE_DELTAS.values()
                 if layout.is_passable((position[0] + delta[0], position[1] + delta[1])))


def all_shortest_paths(layout, start: tuple[int, int], goal: tuple[int, int]) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Return all shortest routes in deterministic coordinate order."""
    queue = deque([start])
    distances = {start: 0}
    while queue:
        current = queue.popleft()
        for target in sorted(_neighbors(layout, current)):
            if target not in distances:
                distances[target] = distances[current] + 1
                queue.append(target)
    if goal not in distances:
        return ()
    result: list[tuple[tuple[int, int], ...]] = []

    def visit(current: tuple[int, int], route: tuple[tuple[int, int], ...]) -> None:
        if current == goal:
            result.append(route)
            return
        for target in sorted(_neighbors(layout, current)):
            if distances.get(target) == distances[current] + 1 and distances[target] <= distances[goal]:
                visit(target, route + (target,))

    visit(start, (start,))
    return tuple(result)


def _directed_edges(route: Sequence[tuple[int, int]]) -> frozenset[tuple[tuple[int, int], tuple[int, int]]]:
    return frozenset(zip(route, route[1:]))


def _undirected_edges(route: Sequence[tuple[int, int]]) -> frozenset[frozenset[tuple[int, int]]]:
    return frozenset(frozenset(edge) for edge in zip(route, route[1:]))


def _graph_bridges(layout) -> frozenset[frozenset[tuple[int, int]]]:
    """Tarjan bridges, kept local so scene selection adds no dependency."""
    timer = 0
    seen: set[tuple[int, int]] = set()
    tin: dict[tuple[int, int], int] = {}
    low: dict[tuple[int, int], int] = {}
    bridges: set[frozenset[tuple[int, int]]] = set()

    def search(node: tuple[int, int], parent: tuple[int, int] | None) -> None:
        nonlocal timer
        seen.add(node)
        tin[node] = low[node] = timer
        timer += 1
        for target in _neighbors(layout, node):
            if target == parent:
                continue
            if target in seen:
                low[node] = min(low[node], tin[target])
                continue
            search(target, node)
            low[node] = min(low[node], low[target])
            if low[target] > tin[node]:
                bridges.add(frozenset((node, target)))

    for node in layout.passable_positions:
        if node not in seen:
            search(node, None)
    return frozenset(bridges)


def _mission_routes(layout, start: tuple[int, int], task) -> tuple[tuple[tuple[int, int], ...], ...]:
    routes = []
    for prefix in all_shortest_paths(layout, start, task.pickup_position):
        for suffix in all_shortest_paths(layout, task.pickup_position, task.delivery_position):
            routes.append(prefix + suffix[1:])
    return tuple(routes)


def geometry_metrics(env: NativeWarehouseEnv) -> dict[str, Any]:
    """Measure conflict in public initial task geometry without an Actor."""
    layout, tasks = env.layout, tuple(env.state.tasks)
    task_routes = [all_shortest_paths(layout, task.pickup_position, task.delivery_position) for task in tasks]
    if any(not routes for routes in task_routes):
        raise ValueError("Unreachable initial task")
    ratios: list[float] = []
    route_pairs: list[tuple[Any, Any, frozenset[frozenset[tuple[int, int]]]]] = []
    for left in task_routes[0]:
        left_edges = _undirected_edges(left)
        for right in task_routes[1]:
            right_edges = _undirected_edges(right)
            shared = left_edges & right_edges
            ratios.append(len(shared) / max(1, min(len(left_edges), len(right_edges))))
            route_pairs.append((left, right, shared))
    bridges = _graph_bridges(layout)
    shared_nodes: set[tuple[int, int]] = set()
    shared_bridge_edges: set[frozenset[tuple[int, int]]] = set()
    shared_intersections: set[tuple[int, int]] = set()
    for left, right, shared in route_pairs:
        shared_bridge_edges.update(shared & bridges)
        nodes = set(left) & set(right)
        shared_nodes.update(nodes)
        shared_intersections.update(node for node in nodes if len(_neighbors(layout, node)) >= 3)

    best_mission: dict[str, Any] | None = None
    minimum_work = math.inf
    for assignment in permutations(range(2)):
        candidates = [_mission_routes(layout, layout.robot_start_positions[role], tasks[assignment[role]])
                      for role in range(2)]
        assignment_work = sum(min(len(route) - 1 for route in routes) for routes in candidates)
        if assignment_work < minimum_work:
            minimum_work, best_mission = assignment_work, None
        if assignment_work != minimum_work:
            continue
        for left in candidates[0]:
            left_directed = _directed_edges(left)
            for right in candidates[1]:
                right_directed = _directed_edges(right)
                same = left_directed & right_directed
                opposite = left_directed & frozenset((after, before) for before, after in right_directed)
                record = {"assignment": list(assignment), "same_direction_edges": len(same),
                          "opposing_edges": len(opposite), "routes": [list(left), list(right)]}
                if best_mission is None or (record["same_direction_edges"] > 0 and record["opposing_edges"] > 0,
                                            record["same_direction_edges"] + record["opposing_edges"]) > (
                                                best_mission["same_direction_edges"] > 0 and best_mission["opposing_edges"] > 0,
                                                best_mission["same_direction_edges"] + best_mission["opposing_edges"]):
                    best_mission = record
    assert best_mission is not None
    checks = {
        "route_overlap_band": min(ratios) >= GEOMETRY["minimum_shared_edge_ratio"] and max(ratios) <= GEOMETRY["maximum_shared_edge_ratio"],
        "shared_bottleneck_or_intersection": bool(shared_bridge_edges or shared_intersections),
        "opposing_and_same_direction_opportunities": best_mission["same_direction_edges"] > 0 and best_mission["opposing_edges"] > 0,
    }
    route_steps = [min(len(route) - 1 for route in routes) for routes in task_routes]
    return {
        "passed": all(checks.values()), "checks": checks,
        "shortest_task_route_shared_edge_ratio_min": min(ratios),
        "shortest_task_route_shared_edge_ratio_max": max(ratios),
        "shared_bridge_edges": [sorted(edge) for edge in sorted(shared_bridge_edges, key=lambda edge: sorted(edge))],
        "shared_intersections": sorted(shared_intersections),
        "same_direction_mission_edges": best_mission["same_direction_edges"],
        "opposing_mission_edges": best_mission["opposing_edges"],
        "initial_task_route_steps": route_steps,
        "initial_joint_work_steps": int(minimum_work),
        "task_endpoints": [{"pickup": task.pickup_position, "delivery": task.delivery_position} for task in tasks],
        "witness": best_mission,
    }


def _task_signature(env: NativeWarehouseEnv) -> str:
    return digest(sorted((task.pickup_position, task.delivery_position) for task in env.state.tasks))


def scan_new_seed_states(manifest: Mapping[str, Any], *, seed_start: int, distinct_count: int,
                         maximum_draws: int, unique_task_geometries: bool = True) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if distinct_count < PROTOCOL["minimum_distinct_new_seed_states"]:
        raise ValueError("At least 60 distinct new seed states are required")
    old_scenes = [scene for scenes in manifest.get("splits", {}).values() for scene in scenes]
    used_seeds = {int(scene["seed"]) for scene in old_scenes}
    old_play_signatures = set()
    probe = NativeWarehouseEnv()
    for scene in manifest.get("splits", {}).get("play", []):
        probe.restore(scene["snapshot"])
        old_play_signatures.add(_task_signature(probe))
    accepted: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()
    seen_observed_states: set[str] = set()
    exclusions = Counter()
    seed = int(seed_start)
    draws = 0
    while len(accepted) < distinct_count and draws < maximum_draws:
        current_seed = seed
        seed += 1
        draws += 1
        if current_seed in used_seeds:
            exclusions["seed_already_in_source_manifest"] += 1
            continue
        env = NativeWarehouseEnv()
        env.reset(seed=current_seed)
        for agent in env.state.agents:
            agent.battery = 100.
        env.state.episode_id = 1
        env._episode_counter = 1
        signature = _task_signature(env)
        observed_signature = digest({"tasks": signature,
                                     "agents": [{"position": agent.position, "heading": agent.heading,
                                                 "battery": agent.battery} for agent in env.state.agents]})
        if signature in old_play_signatures:
            exclusions["old_play_task_geometry"] += 1
            continue
        if observed_signature in seen_observed_states:
            exclusions["duplicate_new_observed_state"] += 1
            continue
        if unique_task_geometries and signature in seen_signatures:
            exclusions["duplicate_new_task_geometry"] += 1
            continue
        seen_signatures.add(signature)
        seen_observed_states.add(observed_signature)
        geometry = geometry_metrics(env)
        accepted.append({"id": f"r4_development_{len(accepted):04d}", "seed": current_seed,
                         "fingerprint": scenario_fingerprint(env), "task_signature": signature,
                         "observed_state_signature": observed_signature,
                         "snapshot": json.loads(canonical(env.snapshot())), "geometry": geometry})
    if len(accepted) < distinct_count:
        raise RuntimeError("Could not produce the pre-registered number of distinct development states")
    return accepted, {"draws": draws, "accepted_distinct_states": len(accepted),
                      "unique_task_geometries_required": unique_task_geometries,
                      "accepted_unique_task_geometries": len(seen_signatures), **dict(exclusions)}


def _legal_player_actions(env: NativeWarehouseEnv) -> tuple[str, ...]:
    player = env.state.by_id("robot_1")
    if not player.active:
        return ("WAIT",)
    legal = ["WAIT"]
    for action, delta in MOVE_DELTAS.items():
        target = (player.position[0] + delta[0], player.position[1] + delta[1])
        if env.layout.is_passable(target):
            legal.append(action)
    return tuple(action for action in ACTIONS if action in legal)


def _dangerous_player_actions(env: NativeWarehouseEnv, actor_action: str) -> tuple[str, ...]:
    return tuple(action for action in _legal_player_actions(env)
                 if env._resolve_motion(env.state, {"robot_1": action, "robot_2": actor_action})[3])


def _actor_action(actor: NumPyNativeActor, env: NativeWarehouseEnv) -> str:
    actions, _ = actor.act(env.observations(), deterministic=True)
    return actions["robot_2"]


def actor_environment(actor: NumPyNativeActor, snapshot: Mapping[str, Any]) -> NativeWarehouseEnv:
    """Build the observation contract encoded in the supplied frozen Actor."""
    if actor.obs_dim == NativeWarehouseEnv().observation_size:
        env: NativeWarehouseEnv = NativeWarehouseEnv()
        env.restore(deepcopy(snapshot))
        return env
    if actor.obs_dim == 197:
        env = OnlinePublicFeedbackEnvironment(reward_config=deepcopy(ONLINE_REWARD), collision_cost=.05, mode="observed")
        env.restore(deepcopy(snapshot))
        if env.observation_size != actor.obs_dim or list(env.feature_names) != actor.metadata.get("feature_names"):
            raise ValueError("Actor observed-history feature contract differs from the r4 audit environment")
        return env
    raise ValueError("Unsupported Actor observation contract for r4 scene audit")


def _distance(env: NativeWarehouseEnv, start: tuple[int, int], goal: tuple[int, int]) -> float:
    return float(shortest_path_distance(start, goal, env.config.map_layout_id))


def compatible_reference_plan(env: NativeWarehouseEnv, actor_action: str) -> dict[str, Any]:
    """Choose a public goal that complements the Actor's actual proposal.

    This is an offline recoverability reference, not a participant model.  It
    reads neither probabilities nor hidden training state.  When two jobs are
    available it treats each possible Actor/job assignment as a hypothesis,
    ranks the hypothesis by progress of the submitted Actor direction, and
    resolves a shared-corridor tie by minimum joint pickup distance.
    """
    participant = env.state.by_id("robot_1")
    actor = env.state.by_id("robot_2")
    charger = env.layout.charger_position
    if not participant.active:
        return {"goal": participant.position, "kind": "inactive", "participant_task_id": None,
                "inferred_actor_task_id": None}
    if participant.carrying_task_id:
        task = env.state.task_by_id(participant.carrying_task_id)
        goal = task.delivery_position
        work = _distance(env, participant.position, goal)
        required = 2 * (work + _distance(env, goal, charger)) + 4
        if participant.battery < min(100., required):
            goal = charger
            kind = "charge_before_delivery"
        else:
            kind = "delivery"
        return {"goal": goal, "kind": kind, "participant_task_id": task.task_id,
                "inferred_actor_task_id": actor.carrying_task_id}

    available = [task for task in env.state.tasks if task.status == "available"]
    inferred_actor_task = None
    participant_task = None
    if available:
        delta = MOVE_DELTAS.get(actor_action, (0, 0))
        actor_next = (actor.position[0] + delta[0], actor.position[1] + delta[1])
        if not env.layout.is_passable(actor_next):
            actor_next = actor.position

        def actor_hypothesis(task) -> tuple[float, float, str]:
            other = min((candidate for candidate in available if candidate.task_id != task.task_id),
                        key=lambda candidate: (_distance(env, participant.position, candidate.pickup_position),
                                               candidate.task_id), default=task)
            before = _distance(env, actor.position, task.pickup_position)
            after = _distance(env, actor_next, task.pickup_position)
            joint_pickup_work = before + _distance(env, participant.position, other.pickup_position)
            return before - after, -joint_pickup_work, task.task_id

        inferred_actor_task = max(available, key=actor_hypothesis)
        alternatives = [task for task in available if task.task_id != inferred_actor_task.task_id]
        participant_task = min(alternatives or available,
                               key=lambda task: (_distance(env, participant.position, task.pickup_position),
                                                 task.task_id))
        goal = participant_task.pickup_position
        work = (_distance(env, participant.position, goal)
                + _distance(env, participant_task.pickup_position, participant_task.delivery_position))
        required = 2 * (work + _distance(env, participant_task.delivery_position, charger)) + 4
        if participant.battery < min(100., required):
            goal = charger
            kind = "charge_before_pickup"
        else:
            kind = "complementary_pickup"
    else:
        goal = charger if participant.battery < 90 else env.layout.robot_start_positions[0]
        kind = "charge_idle" if goal == charger else "idle_home"
    return {"goal": goal, "kind": kind,
            "participant_task_id": None if participant_task is None else participant_task.task_id,
            "inferred_actor_task_id": None if inferred_actor_task is None else inferred_actor_task.task_id}


def compatible_reference_action(env: NativeWarehouseEnv, actor_action: str) -> str:
    """Actor-aware public reference used only to establish scene recoverability."""
    participant = env.state.by_id("robot_1")
    if not participant.active:
        return "WAIT"
    goal = compatible_reference_plan(env, actor_action)["goal"]
    if participant.position == goal:
        return "WAIT"
    ranked = []
    for action in _legal_player_actions(env):
        targets, _, _, collision, _, _ = env._resolve_motion(
            env.state, {"robot_1": action, "robot_2": actor_action})
        target = targets["robot_1"]
        ranked.append(((int(not collision), -_distance(env, target, goal),
                        int(target != participant.position), int(action != "WAIT"),
                        -ACTIONS.index(action)), action))
    return max(ranked)[1]


def update_collision_recovery(pending: list[dict[str, Any]], frame: int, progress: bool) -> int:
    """Resolve each collision at most once inside the inclusive ten-step window."""
    if not progress:
        return 0
    recovered = 0
    for item in pending:
        age = frame - int(item["frame"])
        if not item["resolved"] and 0 <= age <= DYNAMIC["collision_recovery_within_steps"]:
            item["resolved"] = True
            recovered += 1
    return recovered


def simple_action(env: NativeWarehouseEnv, kind: str) -> str:
    if kind == "fixed_yield":
        return partner_action(env, "robot_1", kind, random.Random(0))
    return baseline_action(env, "robot_1", kind)


def run_actor_episode(scene: Mapping[str, Any], actor: NumPyNativeActor, profile: str,
                      continuation_seed: int) -> dict[str, Any]:
    env = actor_environment(actor, scene["snapshot"])
    env.set_rng_state(random.Random(continuation_seed).getstate())
    initial_hash = env.fingerprint()
    pending_collisions: list[dict[str, Any]] = []
    recovered = 0
    steps = opportunities = collisions = actor_submission_frames = actor_action_override_frames = 0
    risky_mass = 0.
    no_progress_streak = longest_no_progress = consecutive_collisions = longest_collisions = 0
    while not env.done:
        actor_action = _actor_action(actor, env)
        dangerous = _dangerous_player_actions(env, actor_action)
        legal = _legal_player_actions(env)
        player_action = (compatible_reference_action(env, actor_action) if profile == "compatible_reference"
                         else simple_action(env, profile))
        before_potential = env.potential()
        before_deliveries = env.state.total_deliveries
        before_pickups = sum(task.status == "carried" for task in env.state.tasks)
        _, _, _, _, info = env.step({"robot_1": player_action, "robot_2": actor_action},
                                    decision_metadata={"policy_action": actor_action, "submitted_action": actor_action,
                                                       "post_policy_overrides": 0})
        if info["requested_actions"]["robot_2"] != actor_action:
            raise AssertionError("Actor action was not submitted unchanged")
        steps += 1
        actor_submission_frames += 1
        actor_action_override_frames += int(info["requested_actions"]["robot_2"] != actor_action)
        opportunities += int(bool(dangerous))
        risky_mass += len(dangerous) / len(legal)
        progress = bool(env.state.total_deliveries > before_deliveries
                        or sum(task.status == "carried" for task in env.state.tasks) > before_pickups
                        or env.potential() > before_potential + 1e-9)
        no_progress_streak = 0 if progress else no_progress_streak + 1
        longest_no_progress = max(longest_no_progress, no_progress_streak)
        collided = bool(info["robot_collision"])
        collisions += int(collided)
        consecutive_collisions = consecutive_collisions + 1 if collided else 0
        longest_collisions = max(longest_collisions, consecutive_collisions)
        if collided:
            pending_collisions.append({"frame": env.state.frame, "resolved": False})
        recovered += update_collision_recovery(pending_collisions, env.state.frame, progress)
    return {"scene_id": scene["id"], "seed": scene["seed"], "profile": profile,
            "continuation_seed": continuation_seed, "initial_fingerprint": initial_hash,
            "steps": steps, "deliveries": env.state.total_deliveries,
            "deliveries_by_robot": {agent.agent_id: agent.deliveries_completed for agent in env.state.agents},
            "collisions": collisions, "shutdowns": env.state.shutdown_count,
            "conflict_opportunity_frames": opportunities,
            "player_risky_action_mass_sum": risky_mass,
            "recovered_collisions_within_10": recovered,
            "longest_consecutive_collisions": longest_collisions,
            "longest_no_progress_streak": longest_no_progress,
            "terminal_consecutive_collisions": consecutive_collisions,
            "terminal_no_progress_streak": no_progress_streak,
            "actor_submission_frames": actor_submission_frames,
            "actor_action_override_frames": actor_action_override_frames,
            "terminal_reason": env.state.terminal_reason}


def _sum(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(sum(row[key] for row in rows))


def dynamic_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    simple = [row for row in rows if row["profile"] in BASELINES]
    reference = [row for row in rows if row["profile"] == "compatible_reference"]
    if len(simple) != len(BASELINES) * len(CONTINUATION_SEEDS) or len(reference) != len(CONTINUATION_SEEDS):
        raise ValueError("Incomplete per-scene dynamic protocol")
    simple_steps = _sum(simple, "steps")
    opportunity = _sum(simple, "conflict_opportunity_frames") / simple_steps
    risky = _sum(simple, "player_risky_action_mass_sum") / simple_steps
    cancellation = _sum(simple, "collisions") / simple_steps
    collisions = _sum(simple, "collisions")
    recovery = _sum(simple, "recovered_collisions_within_10") / collisions if collisions else 1.
    profile_deliveries = {profile: mean(row["deliveries"] for row in simple if row["profile"] == profile)
                          for profile in BASELINES}
    best_profile = max(profile_deliveries, key=lambda key: (profile_deliveries[key], key))
    best_simple = profile_deliveries[best_profile]
    reference_deliveries = mean(row["deliveries"] for row in reference)
    absolute_gap = reference_deliveries - best_simple
    relative_gap = absolute_gap / reference_deliveries if reference_deliveries > 0 else -math.inf
    checks = {
        "conflict_opportunity_25_to_40_percent": DYNAMIC["conflict_opportunity_fraction"][0] <= opportunity <= DYNAMIC["conflict_opportunity_fraction"][1],
        "player_risky_action_mass_10_to_20_percent": DYNAMIC["player_risky_action_mass"][0] <= risky <= DYNAMIC["player_risky_action_mass"][1],
        "collision_cancellation_8_to_18_percent": DYNAMIC["collision_cancellation_fraction"][0] <= cancellation <= DYNAMIC["collision_cancellation_fraction"][1],
        "simple_baseline_gap": absolute_gap >= DYNAMIC["reference_delivery_gap_absolute"] or relative_gap >= DYNAMIC["reference_delivery_gap_relative"],
        "collision_recovery_at_least_90_percent": recovery >= DYNAMIC["minimum_collision_recovery_rate"],
        "no_pathological_zero_delivery_collision_episode": not any(row["deliveries"] == 0 and row["collisions"] > DYNAMIC["reject_zero_delivery_collisions_above"] for row in simple),
        "no_persistent_terminal_collision_deadlock": not any(
            row["terminal_reason"] == DYNAMIC["persistent_deadlock_terminal_reason"]
            and row["terminal_consecutive_collisions"] > DYNAMIC["collision_recovery_within_steps"]
            for row in simple),
        "no_persistent_terminal_no_progress_deadlock": not any(
            row["terminal_reason"] == DYNAMIC["persistent_deadlock_terminal_reason"]
            and row["terminal_no_progress_streak"] > DYNAMIC["collision_recovery_within_steps"]
            for row in simple),
        "actor_actions_submitted_unchanged": sum(row["actor_action_override_frames"] for row in rows) == 0,
    }
    return {"passed": all(checks.values()), "checks": checks,
            "conflict_opportunity_fraction": opportunity, "player_risky_action_mass": risky,
            "collision_cancellation_fraction": cancellation,
            "collision_recovery_within_10_rate": recovery,
            "compatible_reference_mean_deliveries": reference_deliveries,
            "simple_profile_mean_deliveries": profile_deliveries, "best_simple_profile": best_profile,
            "best_simple_mean_deliveries": best_simple, "reference_absolute_delivery_gap": absolute_gap,
            "reference_relative_delivery_gap": relative_gap,
            "max_consecutive_collisions": max(row["longest_consecutive_collisions"] for row in simple),
            "max_no_progress_streak": max(row["longest_no_progress_streak"] for row in simple),
            "max_terminal_consecutive_collisions": max(row["terminal_consecutive_collisions"] for row in simple),
            "max_terminal_no_progress_streak": max(row["terminal_no_progress_streak"] for row in simple),
            "actor_submission_frames": int(_sum(rows, "actor_submission_frames")),
            "actor_action_override_frames": int(_sum(rows, "actor_action_override_frames"))}


def _worker_initialize(actor_path: str) -> None:
    global _WORKER_ACTOR
    _WORKER_ACTOR = NumPyNativeActor(actor_path)


def _evaluate_scene_worker(scene: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if _WORKER_ACTOR is None:
        raise RuntimeError("Dynamic scene worker has no frozen Actor")
    episodes = [run_actor_episode(scene, _WORKER_ACTOR, profile, continuation_seed)
                for profile in (*BASELINES, "compatible_reference")
                for continuation_seed in CONTINUATION_SEEDS]
    return episodes, dynamic_metrics(episodes)


def _relative_difference(left: float, right: float) -> float:
    return abs(left - right) / ((left + right) / 2) if left + right else 0.


def _quality(row: Mapping[str, Any]) -> tuple[float, int]:
    dynamic, geometry = row["dynamic"], row["geometry"]
    score = (abs(dynamic["conflict_opportunity_fraction"] - .325) / .075
             + abs(dynamic["player_risky_action_mass"] - .15) / .05
             + abs(dynamic["collision_cancellation_fraction"] - .13) / .05
             + abs(geometry["shortest_task_route_shared_edge_ratio_min"] - .40) / .15
             - min(dynamic["reference_absolute_delivery_gap"], 4.) / 4.)
    return score, int(row["seed"])


def select_balanced_six(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    passed = sorted((row for row in candidates if row["dynamic"]["passed"]), key=_quality)
    if len(passed) < 6:
        return None

    def pair_compatible(left: int, right: int) -> bool:
        return bool(passed[left]["fingerprint"] != passed[right]["fingerprint"]
            and abs(passed[left]["dynamic"]["compatible_reference_mean_deliveries"]
                    - passed[right]["dynamic"]["compatible_reference_mean_deliveries"])
                <= PAIRING["pair_reference_delivery_difference_max"]
            and abs(passed[left]["dynamic"]["conflict_opportunity_fraction"]
                    - passed[right]["dynamic"]["conflict_opportunity_fraction"])
                <= PAIRING["pair_conflict_opportunity_difference_max"])

    attempts = 0

    def orient(matching: tuple[tuple[int, int], ...]) -> dict[str, Any] | None:
        nonlocal attempts
        attempts += 1
        if len({passed[index]["fingerprint"] for pair in matching for index in pair}) != 6:
            return None
        for orientation in range(8):
            x_indices, y_indices = [], []
            for pair_index, (left, right) in enumerate(matching):
                if orientation & (1 << pair_index):
                    left, right = right, left
                x_indices.append(left); y_indices.append(right)
            workload_x = mean(passed[i]["geometry"]["initial_joint_work_steps"] for i in x_indices)
            workload_y = mean(passed[i]["geometry"]["initial_joint_work_steps"] for i in y_indices)
            conflict_x = mean(passed[i]["dynamic"]["conflict_opportunity_fraction"] for i in x_indices)
            conflict_y = mean(passed[i]["dynamic"]["conflict_opportunity_fraction"] for i in y_indices)
            workload_difference = _relative_difference(workload_x, workload_y)
            conflict_difference = _relative_difference(conflict_x, conflict_y)
            if workload_difference > PAIRING["group_workload_relative_difference_max"] or conflict_difference > PAIRING["group_conflict_relative_difference_max"]:
                continue
            score = (workload_difference + conflict_difference
                     + sum(abs(passed[a]["dynamic"]["conflict_opportunity_fraction"]
                               - passed[b]["dynamic"]["conflict_opportunity_fraction"]) for a, b in matching)
                     + .001 * sum(_quality(passed[i])[0] for i in (*x_indices, *y_indices)))
            return {"score": score, "X": [deepcopy(passed[i]) for i in x_indices],
                    "Y": [deepcopy(passed[i]) for i in y_indices],
                    "pairs": [[passed[x_indices[i]]["id"], passed[y_indices[i]]["id"]] for i in range(3)],
                    "balance": {"X_mean_initial_workload": workload_x, "Y_mean_initial_workload": workload_y,
                                "workload_relative_difference": workload_difference,
                                "X_mean_conflict_opportunity": conflict_x, "Y_mean_conflict_opportunity": conflict_y,
                                "conflict_relative_difference": conflict_difference}}
        return None

    def search(available: tuple[int, ...], matching: tuple[tuple[int, int], ...]) -> dict[str, Any] | None:
        if len(matching) == 3:
            return orient(matching)
        needed = 2 * (3 - len(matching))
        if len(available) < needed:
            return None
        first = available[0]
        partners = sorted((other for other in available[1:] if pair_compatible(first, other)),
                          key=lambda other: (abs(passed[first]["dynamic"]["conflict_opportunity_fraction"]
                                                - passed[other]["dynamic"]["conflict_opportunity_fraction"]),
                                             _quality(passed[other])))
        for other in partners:
            remaining = tuple(index for index in available if index not in (first, other))
            result = search(remaining, matching + ((first, other),))
            if result is not None:
                return result
        return search(available[1:], matching)

    result = search(tuple(range(len(passed))), ())
    if result is not None:
        result["all_dynamic_pass_count"] = len(passed)
        result["pairing_attempts"] = attempts
    return result


def _public_scene(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(row[key]) for key in ("id", "seed", "fingerprint", "task_signature",
                                                 "observed_state_signature", "snapshot")}


def validate_selected_scene_contract(selected: Mapping[str, Any]) -> None:
    """Enforce the exact input contract consumed by the r4 release builder."""
    expected_top = {"version", "actor_sha256", "source_scenario_manifest_sha256", "practice",
                    "X", "Y", "pairs", "balance", "selection_score"}
    expected_scene = {"id", "seed", "fingerprint", "task_signature",
                      "observed_state_signature", "snapshot"}
    if set(selected) != expected_top:
        raise ValueError("Selected-scene manifest has unexpected top-level fields")
    ordered = [selected["practice"], *selected["X"], *selected["Y"]]
    if len(selected["X"]) != 3 or len(selected["Y"]) != 3 or len(ordered) != 7:
        raise ValueError("Selected-scene manifest must contain practice plus three X/Y pairs")
    if any(not isinstance(scene, Mapping) or set(scene) != expected_scene for scene in ordered):
        raise ValueError("Selected-scene manifest has unexpected scene fields")
    if len({scene["seed"] for scene in ordered}) != 7 or len({scene["fingerprint"] for scene in ordered}) != 7:
        raise ValueError("Selected-scene manifest requires seven unique seeds and physical fingerprints")
    expected_pairs = [[selected["X"][index]["id"], selected["Y"][index]["id"]] for index in range(3)]
    if selected["pairs"] != expected_pairs:
        raise ValueError("Selected-scene pairs must correspond exactly to zip(X, Y)")


def deployment_scene_package(manifest: Mapping[str, Any], selected: Mapping[str, Any],
                             actor_sha256: str) -> dict[str, Any]:
    ordered = [selected["practice"], *selected["X"], *selected["Y"]]
    if ordered[0] is None or len(ordered) != 7:
        raise ValueError("Deployment play package requires one practice and six task scenes")
    if len({scene["seed"] for scene in ordered}) != 7 or len({scene["fingerprint"] for scene in ordered}) != 7:
        raise ValueError("Deployment play scenes must have distinct seeds and physical fingerprints")
    play = []
    for index, source in enumerate(ordered):
        scene = deepcopy(source)
        scene["selection_source_id"] = scene.pop("id")
        scene["id"] = f"play_{index:04d}"
        play.append(scene)
    return {"version": VERSION, "configuration": deepcopy(manifest["configuration"]),
            "actor_sha256": actor_sha256, "play": play,
            "indices": {"practice": [0], "X": [1, 2, 3], "Y": [4, 5, 6]},
            "pairs": [[1, 4], [2, 5], [3, 6]],
            "selection_pairs": deepcopy(selected["pairs"]),
            "source_scenario_manifest_sha256": selected["source_scenario_manifest_sha256"]}


def validate_with_online_runtime(package: Mapping[str, Any], actor_path: str | Path,
                                 protocol_path: str | Path) -> dict[str, Any]:
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    actor_sha256, protocol_sha256 = file_hash(actor_path), digest(protocol)
    runtime = OnlineAlignmentRuntime(actor_path, protocol=protocol,
                                     expected_actor_sha256=actor_sha256,
                                     expected_protocol_sha256=protocol_sha256)
    rows = []
    for scene in package["play"]:
        env = runtime.environment(scene)
        before = env.snapshot()
        proposals, decision = runtime.decision(env)
        inference_state_unchanged = env.snapshot() == before
        if not inference_state_unchanged or decision["post_policy_overrides"] != 0:
            raise ValueError("Online runtime reset/inference validation mutated a deployment scene")
        actual_fingerprint = scenario_fingerprint(env)
        branch = runtime.clone(env)
        transition = runtime.step(branch, "WAIT")
        policy_action = transition["policy_actions"]["robot_2"]
        submitted_action = transition["submitted_actions"]["robot_2"]
        rows.append({"id": scene["id"], "seed": scene["seed"], "frame": env.state.frame,
                     "expected_fingerprint": scene["fingerprint"], "actual_fingerprint": actual_fingerprint,
                     "fingerprint_equal": actual_fingerprint == scene["fingerprint"],
                     "public_history_valid": env.public_history()["valid"],
                     "inference_state_unchanged": inference_state_unchanged,
                     "robot_2_policy_action": proposals["robot_2"],
                     "step_policy_action": policy_action, "step_submitted_action": submitted_action,
                     "policy_action_submitted_unchanged": policy_action == submitted_action,
                     "post_policy_overrides": transition["decision"]["post_policy_overrides"]})
    return {"passed": len(rows) == 7 and all(
                row["frame"] == 0 and not row["public_history_valid"]
                and row["fingerprint_equal"] and row["inference_state_unchanged"]
                and row["policy_action_submitted_unchanged"] and row["post_policy_overrides"] == 0
                for row in rows),
            "runtime_signature": runtime.signature, "actor_sha256": actor_sha256,
            "protocol_sha256": protocol_sha256, "scenes": rows}


def audit(scenarios_path: str | Path, output: str | Path, *, actor_path: str | Path | None,
          actor_protocol_path: str | Path | None, seed_start: int, distinct_count: int,
          maximum_draws: int, unique_task_geometries: bool = True, workers: int = 1) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source_bytes = Path(scenarios_path).read_bytes()
    manifest = json.loads(source_bytes)
    actor = NumPyNativeActor(actor_path) if actor_path else None
    source_paths = [Path(__file__), ROOT / "env/warehouse_native/environment.py", ROOT / "env/warehouse_native/scenarios.py",
                    ROOT / "env/warehouse_native/partners.py", ROOT / "env/warehouse_native/policy.py",
                    ROOT / "backend/training/warehouse_native_simple_baselines.py",
                    ROOT / "backend/warehouse_alignment_online_runtime.py"]
    provenance = {"version": VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
                  "protocol": PROTOCOL, "protocol_sha256": digest(PROTOCOL),
                  "source_scenario_manifest": str(Path(scenarios_path).resolve()),
                  "source_scenario_manifest_sha256": sha256(source_bytes).hexdigest(),
                  "actor": None if actor is None else {"path": str(Path(actor_path).resolve()), "sha256": actor.sha256,
                                                        "deterministic": True, "post_policy_overrides": 0},
                  "actor_protocol": None if actor_protocol_path is None else {
                      "path": str(Path(actor_protocol_path).resolve()),
                      "file_sha256": file_hash(actor_protocol_path),
                      "semantic_sha256": digest(json.loads(Path(actor_protocol_path).read_text(encoding="utf-8")))},
                  "source_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in source_paths},
                  "participant_data_read": False, "final_test_rollouts": 0}
    write_json(output / "protocol.json", provenance)
    states, scan = scan_new_seed_states(manifest, seed_start=seed_start, distinct_count=distinct_count,
                                        maximum_draws=maximum_draws,
                                        unique_task_geometries=unique_task_geometries)
    geometry_candidates = [{key: deepcopy(row[key]) for key in ("id", "seed", "fingerprint", "task_signature", "geometry")}
                           for row in states]
    write_json(output / "geometry_candidates.json", {**provenance, "scan": scan,
               "geometry_pass_count": sum(row["geometry"]["passed"] for row in states), "candidates": geometry_candidates})
    report: dict[str, Any] = {**provenance, "scan": scan, "geometry_evaluated": len(states),
                              "geometry_passed": sum(row["geometry"]["passed"] for row in states),
                              "dynamic_evaluated": 0, "dynamic_passed": 0,
                              "status": "geometry_candidates_ready" if actor is None else "dynamic_audit_running"}
    if actor is None:
        report["geometry_shortlist"] = geometry_candidates[:18]
        report["evidence_artifacts"] = {"protocol.json": file_hash(output / "protocol.json"),
                                        "geometry_candidates.json": file_hash(output / "geometry_candidates.json")}
        write_json(output / "report.json", {**report, "elapsed_seconds": time.perf_counter() - started})
        return report

    evaluated = []
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("Dynamic audit workers must be an integer from one to eight")
    with (output / "dynamic_episodes.jsonl").open("x", encoding="utf-8") as stream:
        geometric = [row for row in states if row["geometry"]["passed"]]
        if workers == 1:
            _worker_initialize(str(Path(actor_path).resolve()))
            iterator = map(_evaluate_scene_worker, geometric)
            pool = None
        else:
            pool = ProcessPoolExecutor(max_workers=workers, initializer=_worker_initialize,
                                       initargs=(str(Path(actor_path).resolve()),))
            iterator = pool.map(_evaluate_scene_worker, geometric)
        try:
            for index, (scene, outcome) in enumerate(zip(geometric, iterator)):
                episodes, metrics = outcome
                for episode in episodes:
                    stream.write(canonical(episode) + "\n")
                stream.flush()
                evaluated.append({**{key: deepcopy(scene[key]) for key in ("id", "seed", "fingerprint", "task_signature", "geometry", "snapshot")},
                                  "observed_state_signature": scene["observed_state_signature"],
                                  "dynamic": metrics})
                print(canonical({"completed": index + 1, "total": len(geometric), "seed": scene["seed"],
                                 "dynamic_passed": metrics["passed"]}), flush=True)
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
    selection = select_balanced_six(evaluated)
    selected = None
    deployment_package = None
    if selection is not None:
        selected = {"version": VERSION, "actor_sha256": actor.sha256,
                    "source_scenario_manifest_sha256": sha256(source_bytes).hexdigest(),
                    "practice": None, "X": [_public_scene(row) for row in selection["X"]],
                    "Y": [_public_scene(row) for row in selection["Y"]], "pairs": selection["pairs"],
                    "balance": selection["balance"], "selection_score": selection["score"]}
        selected_ids = {row["id"] for row in (*selection["X"], *selection["Y"])}
        practice_pool = [row for row in states if row["id"] not in selected_ids and not row["geometry"]["passed"]]
        if practice_pool:
            selected["practice"] = _public_scene(min(practice_pool, key=lambda row: (abs(row["geometry"]["initial_joint_work_steps"] - 22), row["seed"])))
        validate_selected_scene_contract(selected)
        package = deployment_scene_package(manifest, selected, actor.sha256)
        if actor_protocol_path is not None:
            package["online_runtime_validation"] = validate_with_online_runtime(
                package, actor_path, actor_protocol_path)
        else:
            package["online_runtime_validation"] = {"passed": False,
                "reason": "actor_protocol_not_supplied; run final audit with --actor-protocol"}
        write_json(output / "selected_scenes.json", selected)
        write_json(output / "deployment_play_scenes.json", package)
        deployment_package = package
    report.update({"dynamic_evaluated": len(evaluated),
                   "dynamic_passed": sum(row["dynamic"]["passed"] for row in evaluated),
                   "actor_submission_frames": sum(row["dynamic"]["actor_submission_frames"] for row in evaluated),
                   "actor_action_override_frames": sum(row["dynamic"]["actor_action_override_frames"] for row in evaluated),
                   "dynamic_workers": workers,
                   "selection": None if selection is None else {key: selection[key] for key in ("pairs", "balance", "score", "pairing_attempts", "all_dynamic_pass_count")},
                   "deployment_runtime_validation": None if deployment_package is None else deployment_package["online_runtime_validation"],
                   "status": ("accepted_six_scene_selection"
                              if deployment_package is not None and deployment_package["online_runtime_validation"].get("passed") is True
                              else "candidate_blocked_runtime_validation"
                              if selection is not None else "candidate_blocked_no_balanced_six"),
                   "scene_metrics": [{key: deepcopy(row[key]) for key in ("id", "seed", "fingerprint", "geometry", "dynamic")} for row in evaluated],
                   "limitations": ["Development simple programs are not participants without explanations.",
                                   "The compatible reference establishes recoverability; it is not an optimality proof.",
                                   "Maximum transient collision and no-progress streaks are diagnostic; only terminal persistent tails can reject a scene.",
                                   "Scene selection does not establish an explanation treatment effect.",
                                   "All scenes share the same compact warehouse topology."]})
    if any(file_hash(path) != provenance["source_sha256"][str(path.relative_to(ROOT))] for path in source_paths):
        raise RuntimeError("Selection source changed during audit")
    if (Path(scenarios_path).read_bytes() != source_bytes or actor.sha256 != file_hash(actor_path)
            or (actor_protocol_path is not None
                and provenance["actor_protocol"]["file_sha256"] != file_hash(actor_protocol_path))):
        raise RuntimeError("Frozen audit input changed during audit")
    report["evidence_artifacts"] = {
        "protocol.json": file_hash(output / "protocol.json"),
        "geometry_candidates.json": file_hash(output / "geometry_candidates.json"),
        "dynamic_episodes.jsonl": file_hash(output / "dynamic_episodes.jsonl"),
        **({"selected_scenes.json": file_hash(output / "selected_scenes.json"),
            "deployment_play_scenes.json": file_hash(output / "deployment_play_scenes.json")}
           if selection is not None else {})}
    write_json(output / "report.json", {**report, "elapsed_seconds": time.perf_counter() - started})
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--actor", type=Path, help="Frozen NumPy Actor; omit for geometry-only selection")
    parser.add_argument("--actor-protocol", type=Path,
                        help="Matching protocol; validates final scenes through OnlineAlignmentRuntime")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=500_000)
    parser.add_argument("--distinct-count", type=int, default=120)
    parser.add_argument("--maximum-draws", type=int, default=20_000)
    parser.add_argument("--allow-repeated-task-geometry", action="store_true",
                        help="For >=300-state expansion: headings must remain distinct; final seven fingerprints still cannot repeat")
    parser.add_argument("--workers", type=int, default=1,
                        help="Independent scene processes (1-8); results retain deterministic scene order")
    args = parser.parse_args(argv)
    if args.actor_protocol is not None and args.actor is None:
        parser.error("--actor-protocol requires --actor")
    report = audit(args.scenarios, args.output, actor_path=args.actor,
                   actor_protocol_path=args.actor_protocol, seed_start=args.seed_start,
                   distinct_count=args.distinct_count, maximum_draws=args.maximum_draws,
                   unique_task_geometries=not args.allow_repeated_task_geometry, workers=args.workers)
    print(canonical({key: report[key] for key in ("status", "geometry_evaluated", "geometry_passed", "dynamic_evaluated", "dynamic_passed")}))


if __name__ == "__main__":
    main()
