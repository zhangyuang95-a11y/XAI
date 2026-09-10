"""Paired, contribution-aware development evaluations; no policy selection hacks."""
from __future__ import annotations

from collections import Counter
import copy
import numpy as np

from env.warehouse.navigation import ACTIONS
from env.warehouse_native.environment import NativeWarehouseEnv
from env.warehouse_native.partners import partner_action
from env.warehouse_native.scenarios import reset_scenario
from .warehouse_native_common import digest, jsonable


def episode(actor, scenario, partner, *, seed=0, keep_trace=False, neural_role=1):
    env = NativeWarehouseEnv()
    reset_scenario(env, scenario)
    actor_rng = np.random.default_rng(np.random.SeedSequence([seed, 101]))
    partner_rng = np.random.default_rng(np.random.SeedSequence([seed, 202]))
    steps = 0
    waits = [0, 0]
    blocked = [0, 0]
    charge_gain = [0., 0.]
    charge_steps = [0, 0]
    counts = Counter()
    max_repeats = 0
    trace = []
    observation_rows = []
    groups = []
    while not (env.state.terminated or env.state.truncated):
        before = env.get_state()
        obs = env.observations()
        if actor == "random":
            proposals = {a.agent_id: ACTIONS[int(actor_rng.integers(5))] for a in before.agents}
        elif actor == "reference":
            proposals = {a.agent_id: partner_action(env, a.agent_id, "skilled", actor_rng) for a in before.agents}
        elif isinstance(actor, str):
            proposals = {a.agent_id: partner_action(env, a.agent_id, actor, actor_rng) for a in before.agents}
        else:
            proposals, _ = actor.act(obs, deterministic=True)
        actions = dict(proposals)
        if partner != "selfplay":
            pid = f"robot_{2 - neural_role}"
            actions[pid] = partner_action(env, pid, partner, partner_rng)
        nid = f"robot_{neural_role + 1}"
        assert actions[nid] == proposals[nid]
        if keep_trace:
            observation_rows.append(obs[nid].tolist())
            groups.append(critical_groups(env, nid))
        _, rewards, terminated, truncated, info = env.step(actions)
        after = env.get_state()
        steps += 1
        executed = info["executed_actions"]
        for i, old in enumerate(before.agents):
            new = after.by_id(old.agent_id)
            waits[i] += int(actions[old.agent_id] == "WAIT")
            blocked[i] += int(actions[old.agent_id] != "WAIT" and executed[old.agent_id] == "WAIT")
            gain = max(0., new.battery - old.battery)
            charge_gain[i] += gain
            charge_steps[i] += int(gain > 0)
        # Recurrent physical/task configuration; time and battery excluded deliberately.
        # This is a diagnostic count, not an error label or training reward.
        key = digest({"positions": [a.position for a in after.agents],
                      "cargo": [a.carrying_task_id for a in after.agents],
                      "tasks": [(t.task_id, t.status) for t in after.tasks],
                      "delivered": after.total_deliveries})
        counts[key] += 1
        max_repeats = max(max_repeats, counts[key])
        if keep_trace:
            trace.append({"before": jsonable(before), "policy_actions": proposals,
                          "submitted_actions": actions, "executed_actions": executed,
                          "after": jsonable(after), "rewards": rewards})
        if terminated or truncated:
            break
    s = env.get_state()
    row = {"scenario_id": scenario["id"], "initial_fingerprint": scenario["fingerprint"],
           "partner": partner, "neural_role": neural_role, "steps": steps,
           "team_deliveries": s.total_deliveries,
           "individual_deliveries": [a.deliveries_completed for a in s.agents],
           "ai_deliveries": s.agents[neural_role].deliveries_completed,
           "ai_active_end": bool(s.agents[neural_role].active),
           "collisions": s.robot_collision_events, "shutdowns": s.shutdown_count,
           "native_score": float(s.user_score), "legacy_score": None,
           "legacy_score_status": "not_replayed_against_original_planner",
           "waits": waits, "blocked_moves": blocked, "charge_gain": charge_gain,
           "charging_steps": charge_steps, "max_repeated_configuration": max_repeats,
           "invalid_moves": s.invalid_move_count, "terminal_reason": s.terminal_reason,
           "nn_action_overrides": 0}
    if keep_trace:
        row.update(trace=trace, observations=observation_rows, groups=groups)
    return row


def critical_groups(env, agent_id):
    """Pre-registered public geometry categories, not Actor goals or actions."""
    from env.warehouse.navigation import shortest_path_distance, MOVE_DELTAS
    s = env.state
    a = s.by_id(agent_id)
    b = next(x for x in s.agents if x.agent_id != agent_id)
    distance = lambda x, y: shortest_path_distance(x, y, env.config.map_layout_id)
    result = []
    # adjacency/occupancy from public layout only
    degree = sum(env.layout.is_passable((a.position[0] + dr, a.position[1] + dc))
                 for dr, dc in MOVE_DELTAS.values())
    if degree <= 2 and distance(a.position, b.position) <= 3:
        result.append("narrow_passage")
    if any(t.status == "available" and distance(a.position, t.pickup_position) <= 4
           and distance(b.position, t.pickup_position) <= 4 for t in s.tasks):
        result.append("shared_pickup")
    charger = env.layout.charger_position
    if (distance(a.position, charger) <= 3 and distance(b.position, charger) <= 3) or (
            min(a.battery, b.battery) <= 30 and max(distance(a.position, charger), distance(b.position, charger)) <= 5):
        result.append("shared_charger")
    return result


def summarize(rows):
    def mean(key):
        return float(np.mean([r[key] for r in rows])) if rows else None
    return {"episodes": len(rows), "mean_team_deliveries": mean("team_deliveries"),
            "mean_ai_deliveries": mean("ai_deliveries"), "ai_active_end_rate": mean("ai_active_end"),
            "mean_native_score": mean("native_score"), "mean_collisions": mean("collisions"),
            "mean_shutdowns": mean("shutdowns"), "mean_steps": mean("steps"),
            "mean_invalid_moves": mean("invalid_moves"),
            "mean_repeated_configuration_max": mean("max_repeated_configuration"),
            "ai_charging_episode_rate": float(np.mean([r["charging_steps"][r["neural_role"]] > 0 for r in rows])) if rows else None,
            "nn_action_overrides": sum(r["nn_action_overrides"] for r in rows)}


def evaluate(actor, scenarios, *, partners=("skilled", "assertive", "noisy"), seed=17000, traces=0):
    rows = []
    for partner in partners:
        for index, scene in enumerate(scenarios):
            rows.append(episode(actor, scene, partner, seed=seed + index,
                                keep_trace=index < traces))
    return {"summary": {p: summarize([r for r in rows if r["partner"] == p]) for p in partners},
            "rows": rows, "deterministic_actor": True, "metric": "mean_team_deliveries"}


def capability(report, reference, random, protocol):
    gate = protocol["candidate_gate"]
    checks = {}
    ratios = {}
    for kind, fraction in (("skilled", gate["minimum_skilled_delivery_ratio_to_reference"]),
                           ("noisy", gate["minimum_noisy_delivery_ratio_to_reference"])):
        got, ref, rnd = report["summary"][kind], reference["summary"][kind], random["summary"][kind]
        denominator = ref["mean_team_deliveries"]
        ratio = got["mean_team_deliveries"] / denominator if denominator and denominator > 0 else None
        ratios[kind] = ratio
        checks[f"{kind}_reference_valid"] = ratio is not None
        checks[f"{kind}_throughput"] = ratio is not None and ratio >= fraction
        checks[f"{kind}_ai_contribution"] = got["mean_ai_deliveries"] >= gate["minimum_ai_deliveries_mean"]
        checks[f"{kind}_ai_survival"] = got["ai_active_end_rate"] >= gate["minimum_ai_active_end_rate"]
        checks[f"{kind}_sample_count"] = got["episodes"] >= gate["episodes_per_partner"]
        rd = rnd["mean_team_deliveries"]
        checks[f"{kind}_random_improvement"] = got["mean_team_deliveries"] >= rd * (1 + gate["minimum_delivery_improvement_over_random"]) and got["mean_team_deliveries"] > rd
    return {"eligible": all(checks.values()), "checks": checks, "reference_ratios": ratios,
            "status": "candidate" if not all(checks.values()) else "capability_passed_not_formal_release"}
