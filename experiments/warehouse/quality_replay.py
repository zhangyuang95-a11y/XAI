"""Independent physical replay of Warehouse v6 quality evidence.

The training runner records a compact action and public-state hash chain for
each quality episode. This module opens the *exported* frozen Actor, reruns the
physical environment, and checks every proposed action, transition, and final
outcome. It has no training dependency and never supplies an action to a live
participant session.
"""
from __future__ import annotations

import base64
import binascii
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor
from env.warehouse.decision_protocol import canonical_sha256
from env.warehouse.domain import participant_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.navigation import ACTIONS
from env.warehouse.public_observation_v6 import public_pre_action_snapshot


TRACE_SCHEMA = "warehouse-quality-trace.v1"
REPLAY_SCHEMA = "warehouse-quality-replay.v1"
ACTION_ENCODING = "base64-u8-a1-times-5-plus-a2"
ROLES = ("robot_1", "robot_2")


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(value: object) -> str:
    return sha256(_json_bytes(value)).hexdigest()


def _environment(seed: int) -> WarehouseMultiAgentEnv:
    environment = WarehouseMultiAgentEnv(participant_study_config())
    environment.reset(seed=seed)
    marked = environment.get_state()
    marked.participant_controlled_agent_id = "robot_1"
    environment.set_state(marked)
    return environment


def replay_episode(actor: FrozenWarehouseV6Actor, row: Mapping,
                   *, scenario_seed: int, action_seed: int) -> dict:
    """Raise if one claimed outcome differs from frozen NN plus physics."""
    trace = row.get("quality_trace")
    if (not isinstance(trace, Mapping)
            or set(trace) != {
                "schema", "actor_sha256", "actor_role_sha256",
                "scenario_seed", "action_seed", "episode_id", "frame_count",
                "action_encoding", "actions_b64", "initial_pre_state_sha256",
                "final_post_state_sha256", "chain_sha256",
            }
            or trace["schema"] != TRACE_SCHEMA
            or trace["actor_sha256"] != actor.artifact_sha256
            or trace["actor_role_sha256"] != actor.role_sha256
            or type(scenario_seed) is not int or type(action_seed) is not int
            or trace["scenario_seed"] != scenario_seed
            or trace["action_seed"] != action_seed
            or trace["action_encoding"] != ACTION_ENCODING
            or type(trace["episode_id"]) is not int
            or type(trace["frame_count"]) is not int
            or trace["frame_count"] <= 0
            or not isinstance(trace["actions_b64"], str)):
        raise ValueError("Warehouse quality trace identity mismatch")
    try:
        encoded = trace["actions_b64"].encode("ascii")
        action_bytes = base64.b64decode(encoded, validate=True)
    except (UnicodeError, binascii.Error) as exc:
        raise ValueError("Warehouse quality action stream malformed") from exc
    if (base64.b64encode(action_bytes) != encoded
            or len(action_bytes) != trace["frame_count"]
            or any(value >= len(ACTIONS) ** 2 for value in action_bytes)):
        raise ValueError("Warehouse quality action stream noncanonical")
    environment = _environment(scenario_seed)
    if environment.state.episode_id != trace["episode_id"]:
        raise ValueError("Warehouse quality episode ID changed")
    chain = sha256(_json_bytes({
        "schema": TRACE_SCHEMA,
        "actor_sha256": actor.artifact_sha256,
        "scenario_seed": scenario_seed,
        "action_seed": action_seed,
        "episode_id": trace["episode_id"],
    })).digest()
    first_pre, last_post = None, None
    for frame_index, action_byte in enumerate(action_bytes):
        if environment.state.terminated or environment.state.truncated:
            raise ValueError("Warehouse quality trace has extra frame")
        frame = int(environment.state.frame)
        if frame != frame_index:
            raise ValueError("Warehouse quality frame sequence changed")
        before = canonical_sha256(public_pre_action_snapshot(environment.state))
        if first_pre is None:
            first_pre = before
        actions, _ = actor.act(environment.state, base_seed=action_seed)
        expected_byte = (len(ACTIONS) * ACTIONS.index(actions[ROLES[0]])
                         + ACTIONS.index(actions[ROLES[1]]))
        if action_byte != expected_byte:
            raise ValueError("Warehouse quality action differs from bound NN")
        _, _, _, _, info = environment.step(actions)
        if (info["proposed_actions"] != actions
                or info["action_override_agents"]):
            raise ValueError("Warehouse quality environment overrode NN action")
        after = canonical_sha256(public_pre_action_snapshot(environment.state))
        chain = sha256(chain + _json_bytes([
            frame, before, action_byte // len(ACTIONS),
            action_byte % len(ACTIONS), after,
        ])).digest()
        last_post = after
    if (not (environment.state.terminated or environment.state.truncated)
            or trace["initial_pre_state_sha256"] != first_pre
            or trace["final_post_state_sha256"] != last_post
            or trace["chain_sha256"] != chain.hex()
            or row.get("seed") != scenario_seed
            or type(row.get("deliveries")) is not int
            or type(row.get("collisions")) is not int
            or type(row.get("shutdown_episode")) is not bool
            or row["deliveries"] != environment.state.total_deliveries
            or row["collisions"] != environment.state.robot_collision_events
            or row["shutdown_episode"] != bool(environment.state.shutdown_count)):
        raise ValueError("Warehouse quality state chain or outcome differs")
    return {"seed": scenario_seed, "action_seed": action_seed,
            "frame_count": len(action_bytes), "chain_sha256": chain.hex(),
            "deliveries": row["deliveries"], "collisions": row["collisions"],
            "shutdown_episode": row["shutdown_episode"]}


def replay_series(*, actor_path: str | Path, actor_sha256: str,
                  role_sha256: Mapping[str, str], rows: Sequence[Mapping],
                  seeds: Sequence[tuple[int, int]]) -> dict:
    """Produce a receipt for a fixed, ordered episode schedule at packaging."""
    claimed = claimed_series(actor_sha256=actor_sha256,
                             role_sha256=role_sha256, rows=rows, seeds=seeds)
    actor = FrozenWarehouseV6Actor.load(
        actor_path, expected_sha256=actor_sha256,
        expected_role_sha256=role_sha256)
    replayed = [replay_episode(actor, row, scenario_seed=seed,
                               action_seed=action_seed)
                for row, (seed, action_seed) in zip(rows, seeds, strict=True)]
    if (claimed["frame_count"] != sum(row["frame_count"] for row in replayed)
            or claimed["chain_list_sha256"] !=
            digest([row["chain_sha256"] for row in replayed])):
        raise ValueError("Warehouse quality replay disagrees with trace inventory")
    return claimed


def claimed_series(*, actor_sha256: str, role_sha256: Mapping[str, str],
                   rows: Sequence[Mapping],
                   seeds: Sequence[tuple[int, int]]) -> dict:
    """Bind a saved series without rerunning it; replay_series certifies it."""
    if (len(rows) != len(seeds) or not rows
            or not isinstance(actor_sha256, str)
            or not isinstance(role_sha256, Mapping)
            or set(role_sha256) != set(ROLES)):
        raise ValueError("Warehouse quality replay schedule mismatch")
    chains, frame_count = [], 0
    for row, (seed, action_seed) in zip(rows, seeds, strict=True):
        if not isinstance(row, Mapping):
            raise ValueError("Warehouse quality row is not an object")
        trace = row.get("quality_trace")
        if (not isinstance(trace, Mapping)
                or set(trace) != {
                    "schema", "actor_sha256", "actor_role_sha256",
                    "scenario_seed", "action_seed", "episode_id", "frame_count",
                    "action_encoding", "actions_b64", "initial_pre_state_sha256",
                    "final_post_state_sha256", "chain_sha256",
                }
                or trace["schema"] != TRACE_SCHEMA
                or trace["actor_sha256"] != actor_sha256
                or trace["actor_role_sha256"] != dict(role_sha256)
                or trace["scenario_seed"] != seed
                or trace["action_seed"] != action_seed
                or trace["action_encoding"] != ACTION_ENCODING
                or type(trace["episode_id"]) is not int
                or trace["episode_id"] < 0
                or type(trace["frame_count"]) is not int
                or trace["frame_count"] <= 0
                or not isinstance(trace["actions_b64"], str)
                or any(not isinstance(trace[name], str) or len(trace[name]) != 64
                       or any(char not in "0123456789abcdef" for char in trace[name])
                       for name in ("initial_pre_state_sha256",
                                    "final_post_state_sha256", "chain_sha256"))
                or row.get("seed") != seed
                or type(row.get("deliveries")) is not int
                or row["deliveries"] < 0
                or type(row.get("collisions")) is not int
                or row["collisions"] < 0
                or type(row.get("shutdown_episode")) is not bool):
            raise ValueError("Warehouse quality trace identity mismatch")
        try:
            encoded = trace["actions_b64"].encode("ascii")
            actions = base64.b64decode(encoded, validate=True)
        except (UnicodeError, binascii.Error) as exc:
            raise ValueError("Warehouse quality action stream malformed") from exc
        if (base64.b64encode(actions) != encoded
                or len(actions) != trace["frame_count"]
                or any(value >= len(ACTIONS) ** 2 for value in actions)):
            raise ValueError("Warehouse quality action stream noncanonical")
        chains.append(trace["chain_sha256"])
        frame_count += len(actions)
    count = len(rows)
    measured = {
        "mean_deliveries": sum(row["deliveries"] for row in rows) / count,
        "mean_collisions": sum(row["collisions"] for row in rows) / count,
        "shutdown_episode_rate": sum(row["shutdown_episode"] for row in rows) / count,
    }
    if not all(math.isfinite(value) for value in measured.values()):
        raise ValueError("Warehouse quality replay produced nonfinite summary")
    return {
        "schema": REPLAY_SCHEMA,
        "actor_sha256": actor_sha256,
        "actor_role_sha256": dict(role_sha256),
        "rows_sha256": digest(rows),
        "schedule_sha256": digest(seeds),
        "episode_count": count,
        "frame_count": frame_count,
        "chain_list_sha256": digest(chains),
        "measured": measured,
        "passed": True,
    }
