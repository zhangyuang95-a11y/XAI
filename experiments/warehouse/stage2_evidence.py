"""Independent evidence checks for exploratory Warehouse v6 stage-two screens.

The frozen Actor is the only policy consulted during a program trace replay.
These checks neither train a policy nor provide actions to a participant task.
Include this module in each candidate's immutable source snapshot before a
stage-two script lock is sealed.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor
from env.warehouse.decision_protocol import canonical_sha256, independent_agent_seed
from env.warehouse.domain import participant_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import AGENTS, action_masks
from env.warehouse.public_observation_v6 import (
    encode_public_observation, public_pre_action_snapshot,
    rebuild_public_pre_action_state,
)


SPLITS = ("fit", "calibration", "eligibility", "validation")


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _read_object(path: Path) -> dict:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"Warehouse stage-two evidence is not an object: {path}")
    return value


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def _environment(seed: int) -> WarehouseMultiAgentEnv:
    env = WarehouseMultiAgentEnv(participant_study_config())
    env.reset(seed=seed)
    marked = env.get_state()
    marked.participant_controlled_agent_id = AGENTS[0]
    env.set_state(marked)
    return env


def replay_program_split_trace(
        actor: FrozenWarehouseV6Actor, trace_path: str | Path, *,
        split: str, seeds: Sequence[int], expected_sha256: str) -> dict:
    """Physically replay every row of one saved four-purpose NN trajectory.

    The saved rows are evidence claims, never inputs to the Actor. The verifier
    reconstructs the environment independently and checks the inputs derived
    from both the actual pre-state and the serialized public pre-state.
    """
    if (split not in SPLITS or not seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int for seed in seeds)):
        raise ValueError("Warehouse program split schedule is invalid")
    path = Path(trace_path)
    raw = path.read_bytes()
    digest = sha256(raw).hexdigest()
    if (digest != expected_sha256 or not raw.endswith(b"\n")
            or path.name != f"{split}.jsonl"):
        raise ValueError("Warehouse program split trace bytes differ")
    try:
        lines = [json.loads(line) for line in raw.splitlines()]
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Warehouse program split trace is malformed") from exc
    cursor = 0
    for seed in seeds:
        env = _environment(seed)
        while not (env.state.terminated or env.state.truncated):
            if cursor >= len(lines):
                raise ValueError("Warehouse program trace has a missing frame")
            row = lines[cursor]
            if (not isinstance(row, Mapping)
                    or set(row) != {"base_seed", "episode_id", "frame",
                                        "pre_state_public", "pre_state_public_sha256",
                                        "post_state_public", "post_state_public_sha256",
                                        "actor_sha256", "roles"}
                    or any(type(row[key]) is not int for key in
                           ("base_seed", "episode_id", "frame"))
                    or (row["base_seed"], row["episode_id"], row["frame"]) !=
                    (seed, int(env.state.episode_id), int(env.state.frame))
                    or row["actor_sha256"] != actor.artifact_sha256
                    or not isinstance(row["roles"], Mapping)
                    or set(row["roles"]) != set(AGENTS)):
                raise ValueError("Warehouse program trace frame identity differs")
            pre = public_pre_action_snapshot(env.state)
            if (row["pre_state_public"] != pre
                    or row["pre_state_public_sha256"] != canonical_sha256(pre)):
                raise ValueError("Warehouse program trace public pre-state differs")
            rebuilt = rebuild_public_pre_action_state(row["pre_state_public"])
            obs, masks = encode_public_observation(env.state), action_masks(env.state)
            if (not np.array_equal(obs, encode_public_observation(rebuilt))
                    or not np.array_equal(masks, action_masks(rebuilt))):
                raise ValueError("Warehouse program trace rebuilt NN inputs differ")
            actions, distributions = actor.act(env.state, base_seed=seed)
            for index, role in enumerate(AGENTS):
                claim = row["roles"][role]
                if (not isinstance(claim, Mapping)
                        or set(claim) != {"obs", "mask", "sampled_action",
                                          "nn_probabilities", "rng_seed",
                                          "actor_role_sha256"}):
                    raise ValueError("Warehouse program trace role row is incomplete")
                role_obs = np.asarray(claim["obs"], dtype=np.float32)
                role_mask = np.asarray(claim["mask"], dtype=np.float32)
                probabilities = np.asarray(claim["nn_probabilities"], dtype=np.float64)
                rng_seed = independent_agent_seed(
                    base_seed=seed, episode_id=int(env.state.episode_id),
                    frame=int(env.state.frame), agent_id=role)
                if (not np.array_equal(role_obs, obs[index])
                        or not np.array_equal(role_mask, masks[index])
                        or probabilities.shape != (len(ACTIONS),)
                        or not np.isfinite(probabilities).all()
                        or not np.allclose(probabilities,
                                           distributions[role].probabilities,
                                           rtol=0.0, atol=1e-7)
                        or claim["sampled_action"] != actions[role]
                        or type(claim["rng_seed"]) is not int
                        or claim["rng_seed"] != rng_seed
                        or claim["actor_role_sha256"] != actor.role_sha256[role]
                        or role_mask[ACTIONS.index(actions[role])] <= .5):
                    raise ValueError("Warehouse program trace NN action or input differs")
            _, _, _, _, info = env.step(actions)
            if (info["proposed_actions"] != actions
                    or info["action_override_agents"]):
                raise ValueError("Warehouse program trace action was rewritten")
            post = public_pre_action_snapshot(env.state)
            if (row["post_state_public"] != post
                    or row["post_state_public_sha256"] != canonical_sha256(post)):
                raise ValueError("Warehouse program trace public post-state differs")
            cursor += 1
    if cursor != len(lines):
        raise ValueError("Warehouse program trace has an extra frame")
    return {"schema": "warehouse-stage2-program-physical-replay.v1",
            "split": split, "actor_sha256": actor.artifact_sha256,
            "actor_role_sha256": dict(actor.role_sha256),
            "trace_sha256": digest, "schedule_sha256": _digest(list(seeds)),
            "episode_count": len(seeds), "frame_count": cursor,
            "independent_physical_replay_equal": True}


def verify_pressure_receipt_identity(
        directory: str | Path, *, runner_script: str | Path,
        common_script: str | Path, prepare_script: str | Path,
        quality_script: str | Path, seed_plan_path: str | Path,
        expected_actor_sha256: str,
        expected_role_sha256: Mapping[str, str],
        expected_source_closure_sha256: str,
        expected_schedule_sha256: str) -> dict:
    """Rebind a passed pressure result to the exact sealed executable bytes.

    A downstream program precheck must call this before sealing its own lock.
    This helper does not reset an environment or infer pressure quality from a
    result alone; the pressure screen's raw/replay checks remain separate.
    """
    root = Path(directory)
    config_path = root / "pressure-predeclared.json"
    lock_path = root / "pressure-script-lock.json"
    result_path = root / "pressure-result.json"
    raw_path = root / "pressure-episodes.jsonl"
    replay_path = root / "pressure-replay.jsonl"
    config, lock, result = (_read_object(path) for path in
                            (config_path, lock_path, result_path))
    scripts = {"script_sha256": _sha(Path(runner_script)),
               "common_sha256": _sha(Path(common_script)),
               "prepare_script_sha256": _sha(Path(prepare_script)),
               "quality_integrity_script_sha256": _sha(Path(quality_script))}
    expected_roles = dict(expected_role_sha256)
    if set(expected_roles) != set(AGENTS):
        raise ValueError("Warehouse pressure Actor role identity is incomplete")
    pre_sha, lock_sha = _sha(config_path), _sha(lock_path)
    plan_sha = _sha(Path(seed_plan_path))
    schedule = config.get("schedule")
    if (not isinstance(schedule, list) or not schedule
            or _digest(schedule) != expected_schedule_sha256
            or not str(config.get("schema", "")).endswith(
                "-pressure-predeclared.v1")
            or config.get("status") != "sealed_unregistered_not_run"
            or config.get("schedule_sha256") != expected_schedule_sha256
            or config.get("stage2_seed_plan_sha256") != plan_sha
            or config.get("stage2_common_sha256") != scripts["common_sha256"]
            or config.get("stage2_prepare_script_sha256") !=
            scripts["prepare_script_sha256"]
            or config.get("quality_integrity_script_sha256") !=
            scripts["quality_integrity_script_sha256"]
            or config.get("actor_sha256") != expected_actor_sha256
            or config.get("actor_role_sha256") != expected_roles
            or config.get("source_closure_sha256") !=
            expected_source_closure_sha256
            or not str(lock.get("schema", "")).endswith(
                "-pressure-script-lock.v1")
            or lock.get("stage") != "pressure"
            or lock.get("script_sha256") != scripts["script_sha256"]
            or lock.get("common_sha256") != scripts["common_sha256"]
            or lock.get("prepare_script_sha256") !=
            scripts["prepare_script_sha256"]
            or lock.get("quality_integrity_script_sha256") !=
            scripts["quality_integrity_script_sha256"]
            or lock.get("predeclared_sha256") != pre_sha
            or lock.get("seed_plan_sha256") != plan_sha
            or lock.get("schedule_sha256") != expected_schedule_sha256
            or lock.get("source_closure_sha256") !=
            expected_source_closure_sha256
            or lock.get("actor_sha256") != expected_actor_sha256
            or lock.get("actor_role_sha256") != expected_roles
            or lock.get("python_command") !=
            f"python3 -I {Path(runner_script).resolve()}"
            or lock.get("required_cwd") != config.get("required_cwd")
            or not str(result.get("schema", "")).endswith(
                "-pressure-result.v1")
            or result.get("status") !=
            "unregistered_frozen_bc_pressure_complete"
            or result.get("predeclared_sha256") != pre_sha
            or result.get("script_lock_sha256") != lock_sha
            or result.get("script_sha256") != scripts["script_sha256"]
            or result.get("actor_sha256") != expected_actor_sha256
            or result.get("actor_role_sha256") != expected_roles
            or result.get("source_closure_sha256") !=
            expected_source_closure_sha256
            or result.get("pressure_schedule_sha256") !=
            expected_schedule_sha256
            or result.get("raw_jsonl_sha256") != _sha(raw_path)
            or result.get("replay_jsonl_sha256") != _sha(replay_path)
            or result.get("pressure_passed") is not True
            or result.get("registered_or_sealed_seeds_used") is not False
            or result.get("program_or_ppo_used") is not False
            or not isinstance(result.get("pressure_gates"), Mapping)
            or not result["pressure_gates"]
            or any(value is not True for value in
                   result["pressure_gates"].values())):
        raise ValueError("Warehouse pressure result/script-lock binding differs")
    return {"pressure_result_sha256": _sha(result_path),
            "pressure_predeclared_sha256": pre_sha,
            "pressure_script_lock_sha256": lock_sha,
            "pressure_runner_script_sha256": scripts["script_sha256"]}
