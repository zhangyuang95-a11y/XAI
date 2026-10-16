"""Independent evidence checks for exploratory Warehouse v6 stage-two screens.

The frozen Actor is the only policy consulted during a program trace replay.
These checks neither train a policy nor provide actions to a participant task.
Include this module in each candidate's immutable source snapshot before a
stage-two script lock is sealed.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np

from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor
from env.warehouse.public_action_features_v614 import validate_study_action_physics
from env.warehouse.decision_protocol import canonical_sha256, independent_agent_seed
from env.warehouse.domain import participant_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import AGENTS, action_masks
from env.warehouse.public_observation_v6 import (
    encode_public_observation, public_pre_action_snapshot,
    rebuild_public_pre_action_state,
)
from experiments.warehouse.protocol import (
    INITIAL_RELEASE_ID, REGISTERED_BC_SPEC, REGISTERED_BC_SPEC_SHA256,
    bc_critic_reset_schedule, bc_data_seed_schedule, bc_stage_receipt_shape,
    bc_report_valid,
)
from env.warehouse.public_action_features_v614 import (
    ACTION_PHYSICS_CONFIG, ACTION_PHYSICS_CONFIG_SHA256,
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
    config = participant_study_config()
    validate_study_action_physics(config)
    env = WarehouseMultiAgentEnv(config)
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


def _v614_lines(path: Path, count: int) -> tuple[list[dict], bytes]:
    """Read one immutable JSONL evidence file without touching an environment."""
    data = path.read_bytes()
    if not data.endswith(b"\n") or len(data.splitlines()) != count:
        raise ValueError(f"Warehouse v6.14 evidence row count differs: {path}")
    try:
        rows = [json.loads(line) for line in data.splitlines()]
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Warehouse v6.14 evidence JSONL is malformed: {path}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Warehouse v6.14 evidence row is not an object: {path}")
    return rows, data


def _v614_schedule(rows: object, *, first: int, count: int,
                   streams: range) -> list[dict]:
    if not isinstance(rows, list):
        raise ValueError("Warehouse v6.14 schedule is missing")
    expected = [{"task": task, "base_seed": base, "stream": stream,
                 "scenario_seed": base * 10 + task,
                 "action_seed": base * 10 + task + stream * 1_000_000,
                 "seed": base * 10 + task}
                for task in (1, 2, 3)
                for base in range(first, first + count)
                for stream in streams]
    if rows != expected:
        raise ValueError("Warehouse v6.14 reserved schedule differs")
    return expected


def _v614_outcome_rows(raw: list[dict], replay: list[dict],
                        schedule: list[dict], replay_sha256: str,
                        label: str) -> dict:
    if len(raw) != len(schedule) or len(replay) != len(schedule):
        raise ValueError(f"Warehouse v6.14 {label} evidence count differs")
    deliveries, collisions, shutdowns = [], [], []
    for index, expected in enumerate(schedule):
        actual, checked = raw[index], replay[index]
        for row in (actual, checked):
            if any(row.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Warehouse v6.14 {label} row schedule differs")
            if (type(row.get("deliveries")) not in (int, float)
                    or type(row.get("collisions")) not in (int, float)
                    or not math.isfinite(row["deliveries"])
                    or not math.isfinite(row["collisions"])
                    or row["deliveries"] < 0 or row["collisions"] < 0
                    or type(row.get("shutdown_episode")) is not bool):
                raise ValueError(f"Warehouse v6.14 {label} outcome is malformed")
        if (any(actual[key] != checked[key] for key in
                ("deliveries", "collisions", "shutdown_episode"))
                or checked.get("independent_physical_replay") is not True
                or checked.get("replay_source_sha256") != replay_sha256
                or not isinstance(actual.get("quality_trace"), Mapping)
                or actual["quality_trace"].get("chain_sha256") !=
                   checked.get("chain_sha256")
                or not isinstance(checked.get("chain_sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", checked["chain_sha256"])
                   is None):
            raise ValueError(f"Warehouse v6.14 {label} physical replay differs")
        deliveries.append(float(actual["deliveries"]))
        collisions.append(float(actual["collisions"]))
        shutdowns.append(int(actual["shutdown_episode"]))
    n = len(schedule)
    return {"mean_deliveries": sum(deliveries) / n,
            "mean_collisions": sum(collisions) / n,
            "shutdown_episode_rate": sum(shutdowns) / n}


def _v614_measurement(result: Mapping, label: str, measured: dict,
                       threshold: Mapping) -> None:
    claimed = result.get(f"{label}_measured")
    gates = result.get(f"{label}_gates")
    if (not isinstance(claimed, Mapping)
            or any(type(claimed.get(key)) not in (int, float)
                   or abs(float(claimed[key]) - value) > 1e-9
                   for key, value in measured.items())
            or threshold != {"mean_deliveries_min": 5.5,
                             "mean_collisions_max": 6.0,
                             "shutdown_episode_rate_max": 0.05}
            or measured["mean_deliveries"] < 5.5
            or measured["mean_collisions"] > 6.0
            or measured["shutdown_episode_rate"] > 0.05
            or not isinstance(gates, Mapping)
            or set(gates) != set(measured)
            or any(value is not True for value in gates.values())
            or result.get(f"{label}_passed") is not True):
        raise ValueError(f"Warehouse v6.14 {label} screen failed or drifted")


def verify_stage1_pressure_receipt_identity(
        package_root: str | Path, *, expected_candidate_id: str,
        expected_pair_seed: int, expected_actor_sha256: str,
        expected_role_sha256: Mapping[str, str],
        expected_source_closure_sha256: str,
        expected_training_recipe_sha256: str,
        expected_quality_schedule_sha256: str,
        expected_pressure_schedule_sha256: str) -> dict:
    """Check a locked single-script Stage 1 pressure result before program.

    This verifier is deliberately independent of the old multi-script v6.9
    receipt. It reads already-fsynced evidence bytes, performs no reset, and
    cannot qualify a candidate from a pressure summary alone. The runner
    separately checks the full reset sequence after program collection.
    """
    root = Path(package_root).resolve()
    paths = {name: root / name for name in (
        "predeclared-run.json", "script-lock.json", "source-snapshot.json",
        "training-recipe.json", "stage1-authorization.json",
        "historical-seed-audit.json", "seed-inventory-independent.json",
        "bc-training-receipt.json", "quality-result.json",
        "pressure-result.json", "quality-episodes.jsonl",
        "quality-replay.jsonl", "quality-replay-receipt.json",
        "pressure-episodes.jsonl", "pressure-replay-receipt.json",
        "pressure-replay.jsonl", "reset-audit.jsonl")}
    pre, lock, snapshot, authorization, historical, seed_inventory, bc, quality, pressure = (
        _read_object(paths[name]) for name in (
            "predeclared-run.json", "script-lock.json", "source-snapshot.json",
            "stage1-authorization.json", "historical-seed-audit.json",
            "seed-inventory-independent.json", "bc-training-receipt.json",
            "quality-result.json", "pressure-result.json"))
    roles = dict(expected_role_sha256)
    if (expected_pair_seed != 12469 or set(roles) != set(AGENTS)
            or any(not isinstance(value, str) or len(value) != 64
                   for value in roles.values())
            or not isinstance(expected_candidate_id, str)
            or not expected_candidate_id):
        raise ValueError("Warehouse v6.14 Stage 1 expected identity is invalid")
    source_files = snapshot.get("source_files_sha256")
    source_root = root / "source"
    if (snapshot.get("schema") != "warehouse-v614-source-snapshot.v1"
            or not isinstance(source_files, Mapping) or not source_files
            or _digest(source_files) != expected_source_closure_sha256
            or snapshot.get("source_closure_sha256") !=
               expected_source_closure_sha256):
        raise ValueError("Warehouse v6.14 source closure manifest differs")
    for relative, digest in source_files.items():
        if (not isinstance(relative, str) or not isinstance(digest, str)
                or not (source_root / relative).resolve().is_relative_to(source_root)
                or _sha(source_root / relative) != digest):
            raise ValueError("Warehouse v6.14 source snapshot bytes differ")
    copied = {path.relative_to(source_root).as_posix()
              for path in source_root.rglob("*") if path.is_file()}
    if copied != set(source_files):
        raise ValueError("Warehouse v6.14 source snapshot inventory differs")
    recipe_sha = _sha(paths["training-recipe.json"])
    if (recipe_sha != expected_training_recipe_sha256
            or recipe_sha != REGISTERED_BC_SPEC[
                "executable_recipe_file_sha256"]
            or recipe_sha != _sha(source_root /
                                 "configs/warehouse_v614_exploratory_recipe.json")):
        raise ValueError("Warehouse v6.14 executable recipe bytes differ")
    script_path = Path(pre.get("lock_script_path", "")).resolve()
    if (not script_path.is_relative_to(root)
            or not script_path.is_file()):
        raise ValueError("Warehouse v6.14 locked script is unavailable")
    script_sha = _sha(script_path)
    pre_sha, lock_sha = _sha(paths["predeclared-run.json"]), _sha(paths["script-lock.json"])
    auth_sha, bc_sha, quality_sha = (
        _sha(paths[name]) for name in (
            "stage1-authorization.json", "bc-training-receipt.json",
            "quality-result.json"))
    common = {"candidate_id": expected_candidate_id,
              "pair_seed": expected_pair_seed,
              "source_closure_sha256": expected_source_closure_sha256,
              "action_physics_config_sha256": ACTION_PHYSICS_CONFIG_SHA256}
    for name, record in (("predeclared", pre), ("script-lock", lock),
                         ("authorization", authorization),
                         ("quality", quality), ("pressure", pressure)):
        if any(record.get(key) != value for key, value in common.items()):
            raise ValueError(f"Warehouse v6.14 {name} identity differs")
    if (pre.get("schema") != "warehouse-v614-12469-predeclared-run.v1"
            or lock.get("schema") != "warehouse-v614-12469-script-lock.v1"
            or authorization.get("authorization_kind") !=
               "warehouse-v614-stage1-locked-run-v1"
            or quality.get("schema") !=
               "warehouse-v614-stage1-quality-result.v1"
            or pressure.get("schema") !=
               "warehouse-v614-stage1-pressure-result.v1"
            or pressure.get("status") !=
               "unregistered_frozen_bc_pressure_complete"
            or quality.get("status") !=
               "unregistered_frozen_bc_quality_complete"
            or pressure.get("registered_or_sealed_seeds_used") is not False
            or quality.get("registered_or_sealed_seeds_used") is not False
            or pressure.get("program_or_ppo_used_at_pressure") is not False
            or quality.get("program_or_ppo_used_at_pressure") is not False):
        raise ValueError("Warehouse v6.14 Stage 1 schema or status differs")
    for record in (pre, lock):
        if (record.get("source_snapshot_sha256") !=
                _sha(paths["source-snapshot.json"])
                or record.get("training_recipe_file_sha256") != recipe_sha
                or record.get("lock_script_path") != str(script_path)
                or record.get("lock_script_sha256") != script_sha):
            raise ValueError("Warehouse v6.14 Stage 1 declaration differs")
    if (lock.get("predeclared_sha256") != pre_sha
            or lock.get("authorization_receipt_sha256") != auth_sha
            or lock.get("historical_seed_audit_sha256") !=
               _sha(paths["historical-seed-audit.json"])
            or lock.get("independent_seed_inventory_sha256") !=
               _sha(paths["seed-inventory-independent.json"])
            or lock.get("python_command") != f"python3 -I -B {script_path}"
            or lock.get("required_cwd") != pre.get("required_cwd")
            or authorization.get("lock_script_path") != str(script_path)
            or authorization.get("lock_script_sha256") != script_sha
            or authorization.get("bc_recipe_sha256") != REGISTERED_BC_SPEC_SHA256
            or authorization.get("executable_recipe_file_sha256") != recipe_sha
            or authorization.get("bc_data_seed_schedule_sha256") !=
               _digest(bc_data_seed_schedule(expected_pair_seed))
            or authorization.get("run_spec_sha256") !=
               _digest(pre.get("run_spec"))
            or pre.get("bc_recipe_sha256") != REGISTERED_BC_SPEC_SHA256):
        raise ValueError("Warehouse v6.14 Stage 1 authorization differs")
    if (historical.get("schema") !=
            "warehouse-v614-12469-historical-seed-audit.v1"
            or historical.get("candidate_pair_seed") != expected_pair_seed
            or historical.get("candidate_schedule_sha256") != _digest({
                "bc": pre.get("bc_data_seed_schedule"),
                "critic": pre.get("bc_critic_reset_schedule"),
                "quality": pre.get("first_quality_schedule"),
                "pressure": pre.get("reserved_pressure_schedule"),
                "program": pre.get("reserved_post_bc_program_schedule"),
                "ppo_final_program": pre.get(
                    "reserved_ppo_final_program_schedule")})
            or any(historical.get(key) != 0 for key in (
                "candidate_historical_overlap_count",
                "candidate_registered_sealed_overlap_count",
                "candidate_internal_overlap_count"))
            or historical.get("independent_auditor_receipt_sha256") !=
               _sha(paths["seed-inventory-independent.json"])
            or seed_inventory.get("all_candidate_groups_disjoint_from_protected")
               is not True
            or seed_inventory.get("all_candidate_groups_pairwise_disjoint")
               is not True):
        raise ValueError("Warehouse v6.14 independent seed audit differs")
    for name, record in (("quality", quality), ("pressure", pressure)):
        if (record.get("predeclared_sha256") != pre_sha
                or record.get("script_lock_sha256") != lock_sha
                or record.get("authorization_receipt_sha256") != auth_sha
                or record.get("runner_script_sha256") != script_sha
                or record.get("training_recipe_sha256") != recipe_sha
                or record.get("actor_sha256") != expected_actor_sha256
                or record.get("actor_role_sha256") != roles
                or record.get("bc_training_receipt_sha256") != bc_sha):
            raise ValueError(f"Warehouse v6.14 {name} provenance differs")
    if (pressure.get("quality_result_sha256") != quality_sha
            or bc.get("final_bc_actor_sha256") != expected_actor_sha256
            or bc.get("final_bc_role_sha256") != roles
            or not bc_report_valid(bc.get("bc_report"), expected_pair_seed,
                                   expected_source_closure_sha256)):
        raise ValueError("Warehouse v6.14 BC or quality prerequisite differs")
    internal_receipt = bc["bc_report"]["final_bc_internal_precheck_receipt"]
    internal_path = Path(internal_receipt["path"]).resolve()
    if (not internal_path.is_relative_to(root)
            or not internal_path.is_file()
            or _sha(internal_path) != internal_receipt["sha256"]):
        raise ValueError("Warehouse v6.14 final BC precheck bytes differ")
    actor_path = Path(bc.get("final_bc_actor_path", "")).resolve()
    if (not actor_path.is_relative_to(root) or not actor_path.is_file()
            or _sha(actor_path) != expected_actor_sha256):
        raise ValueError("Warehouse v6.14 frozen BC Actor bytes differ")
    quality_schedule = _v614_schedule(
        pre.get("first_quality_schedule"), first=1_240_500, count=32,
        streams=range(1))
    pressure_schedule = _v614_schedule(
        pre.get("reserved_pressure_schedule"), first=1_240_200, count=32,
        streams=range(3))
    if (_digest(quality_schedule) != expected_quality_schedule_sha256
            or pre.get("quality_schedule_sha256") !=
               expected_quality_schedule_sha256
            or quality.get("quality_schedule_sha256") !=
               expected_quality_schedule_sha256
            or _digest(pressure_schedule) != expected_pressure_schedule_sha256
            or pre.get("pressure_schedule_sha256") !=
               expected_pressure_schedule_sha256
            or pressure.get("pressure_schedule_sha256") !=
               expected_pressure_schedule_sha256
            or pressure.get("pressure_episode_count") != 288):
        raise ValueError("Warehouse v6.14 quality or pressure seed lock differs")
    if quality.get("quality_episode_count") != 96:
        raise ValueError("Warehouse v6.14 quality episode count differs")
    expected_bc_schedule = bc_data_seed_schedule(expected_pair_seed)
    expected_critic = bc_critic_reset_schedule(expected_pair_seed,
                                               INITIAL_RELEASE_ID)
    run_spec = pre["run_spec"]
    if (not isinstance(run_spec, Mapping)
            or pre.get("bc_data_seed_schedule") != expected_bc_schedule
            or pre.get("bc_data_seed_schedule_sha256") !=
               _digest(expected_bc_schedule)
            or pre.get("bc_critic_reset_schedule") != expected_critic
            or pre.get("bc_critic_reset_schedule_sha256") !=
               _digest(expected_critic)
            or run_spec.get("bc_data_seed_schedule") != expected_bc_schedule
            or run_spec.get("bc_critic_reset_schedule") != expected_critic
            or run_spec.get("source_closure_sha256") !=
               expected_source_closure_sha256
            or run_spec.get("stage1_expected_reset_sequence") !=
               pre.get("stage1_expected_reset_sequence")
            or run_spec.get("reset_audit_journal_path") !=
               str(paths["reset-audit.jsonl"])):
        raise ValueError("Warehouse v6.14 runner Stage 1 spec differs")
    replay_helper = source_root / "experiments/warehouse/quality_replay.py"
    replay_sha = _sha(replay_helper)
    from experiments.warehouse.quality_replay import claimed_series
    for name, record, schedule in (("quality", quality, quality_schedule),
                                   ("pressure", pressure, pressure_schedule)):
        raw_path, replay_path = paths[f"{name}-episodes.jsonl"], paths[f"{name}-replay.jsonl"]
        raw, _ = _v614_lines(raw_path, len(schedule))
        replay, _ = _v614_lines(replay_path, len(schedule))
        if (record.get("raw_jsonl_sha256") != _sha(raw_path)
                or record.get("replay_jsonl_sha256") != _sha(replay_path)
                or record.get("replay_source_sha256") != replay_sha):
            raise ValueError(f"Warehouse v6.14 {name} evidence bytes differ")
        measured = _v614_outcome_rows(raw, replay, schedule, replay_sha, name)
        claimed = claimed_series(
            actor_sha256=expected_actor_sha256, role_sha256=roles,
            rows=raw, seeds=[(row["seed"], row["action_seed"])
                             for row in schedule])
        replay_receipt = _read_object(paths[f"{name}-replay-receipt.json"])
        if (replay_receipt != {
                **claimed, "independent_physical_replay": True,
                "replay_source_sha256": replay_sha,
                "raw_jsonl_sha256": _sha(raw_path),
                "replay_jsonl_sha256": _sha(replay_path)}
                or record.get("physical_replay_receipt_sha256") !=
                   _sha(paths[f"{name}-replay-receipt.json"])
                or claimed["measured"] != measured):
            raise ValueError(f"Warehouse v6.14 {name} replay receipt differs")
        _v614_measurement(record, name, measured,
                           pre.get("first_quality_screen" if name == "quality"
                                   else "pressure_screen"))
    bc_schedule = expected_bc_schedule
    critic = expected_critic
    expected_prefix: list[dict] = []
    for stage, epochs in bc_stage_receipt_shape():
        if epochs is None:
            expected_prefix.extend({"phase": "bc", "seed": seed}
                                   for seed in bc_schedule[stage])
            if stage == "topology_fit_argmax":
                expected_prefix.extend({"phase": "bc", "seed": critic["seed"]}
                                       for _ in range(2))
    for phase, schedule in (("quality", quality_schedule),
                            ("pressure", pressure_schedule)):
        expected_prefix.extend({"phase": phase, "seed": row["seed"]}
                               for row in schedule)
        expected_prefix.extend({"phase": phase, "seed": row["seed"]}
                               for row in schedule)
    external_sequence = pre.get("stage1_expected_reset_sequence")
    external_prefix = expected_prefix[1396:]
    program_schedule = pre.get("reserved_post_bc_program_schedule")
    first = 1_240_300
    expected_program = {}
    for purpose, count in (("fit", 50), ("calibration", 25),
                           ("eligibility", 25), ("validation", 24)):
        expected_program[purpose] = _v614_schedule(
            [{"task": task, "base_seed": base, "stream": 0,
              "scenario_seed": base * 10 + task,
              "action_seed": base * 10 + task, "seed": base * 10 + task}
             for task in (1, 2, 3) for base in range(first, first + count)],
            first=first, count=count, streams=range(1))
        first += count
    program_rows = [{"phase": "program", "seed": row["seed"]}
                    for purpose in ("fit", "calibration", "eligibility",
                                    "validation")
                    for row in expected_program[purpose]]
    if (not isinstance(external_sequence, list)
            or len(expected_prefix) != 2164
            or len(external_sequence) != 1512
            or external_sequence[:768] != external_prefix
            or external_sequence[768:] != program_rows * 2
            or program_schedule != expected_program
            or pre.get("program_schedule_sha256") != _digest(expected_program)
            or _digest(external_sequence) !=
               pre.get("stage1_expected_reset_sequence_sha256")
            or authorization.get("stage1_expected_reset_sequence_sha256") !=
               _digest(external_sequence)
            or lock.get("stage1_expected_reset_sequence_sha256") !=
               _digest(external_sequence)):
        raise ValueError("Warehouse v6.14 full reset sequence lock differs")
    journal, journal_bytes = _v614_lines(paths["reset-audit.jsonl"], 2164)
    for observed, expected in zip(journal, expected_prefix):
        if (observed.get("phase") != expected["phase"]
                or observed.get("seed") != expected["seed"]
                or observed.get("map_layout_id") !=
                   ACTION_PHYSICS_CONFIG["map_layout_id"]
                or observed.get("move_battery_cost") !=
                   ACTION_PHYSICS_CONFIG["move_battery_cost"]
                or observed.get("charge_per_wait") !=
                   ACTION_PHYSICS_CONFIG["charge_per_wait"]
                or type(observed.get("episode_id")) is not int):
            raise ValueError("Warehouse v6.14 reset journal prefix differs")
    quality_prefix = b"\n".join(journal_bytes.splitlines()[:1588]) + b"\n"
    if (quality.get("reset_journal_prefix_sha256") !=
            sha256(quality_prefix).hexdigest()
            or quality.get("reset_journal_prefix_row_count") != 1588
            or quality.get("reset_schedule_prefix_sha256") !=
               _digest(external_prefix[:192])
            or pressure.get("reset_journal_prefix_sha256") !=
               sha256(journal_bytes).hexdigest()
            or pressure.get("reset_journal_prefix_row_count") != 2164
            or pressure.get("reset_schedule_prefix_sha256") !=
               _digest(external_prefix)):
        raise ValueError("Warehouse v6.14 prefix reset evidence differs")
    return {"schema": "warehouse-v614-stage1-pressure-identity.v1",
            "pressure_result_sha256": _sha(paths["pressure-result.json"]),
            "quality_result_sha256": quality_sha,
            "final_bc_actor_sha256": expected_actor_sha256,
            "actor_role_sha256": roles,
            "source_closure_sha256": expected_source_closure_sha256,
            "training_recipe_sha256": recipe_sha,
            "authorization_receipt_sha256": auth_sha,
            "reset_journal_prefix_sha256": sha256(journal_bytes).hexdigest(),
            "reset_journal_prefix_row_count": 2164,
            "independent_physical_replay_equal": True}
