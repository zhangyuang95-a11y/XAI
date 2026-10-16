"""Recheck the complete packaged Warehouse v6 admission chain at runtime.

This verifier never trains, rolls out, or selects an action. It reads immutable,
content-addressed receipts and derives every release gate again from raw rows.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from . import protocol as p


_PHASE_NAMES = (
    "budget.lock.json", "formal.selection.json",
    *(f"formal.training.{seed}.json" for seed in p.FORMAL_PAIR_SEEDS),
    "program.preaudit.json", "final_program_audit.json", "heldout.json",
)


def _digest(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def _sha(value: object) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def _asset_bytes(root: Path, release_id: str, item: Mapping) -> tuple[bytes, str]:
    path, expected = item["path"], item["sha256"]
    prefix = f"releases/{release_id}/assets/"
    if (not isinstance(path, str) or not path.startswith(prefix)
            or not _sha(expected) or path.split("/")[-2] != expected):
        raise ValueError("Warehouse evidence is not content addressed")
    target = (root / path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Warehouse evidence escaped repository")
    raw = target.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError("Warehouse evidence hash changed")
    return raw, expected


def _asset(root: Path, release_id: str, item: Mapping) -> tuple[dict, str]:
    raw, expected = _asset_bytes(root, release_id, item)
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("Warehouse evidence is not a JSON object")
    return result, expected


def _policy_ids(release_id: str) -> dict[str, str]:
    p.sealed_seed_blocks(release_id)
    minor_text = release_id.rsplit(".v6.", 1)[1]
    minor = int(minor_text)
    if str(minor) != minor_text:
        raise ValueError("Warehouse release minor version is noncanonical")
    suffix = "v6" if minor == 0 else f"v6.{minor}"
    return {variant: f"warehouse-{variant}-{suffix}"
            for variant in p.VARIANTS}


def _training_assets(root: Path, release_id: str, index: Mapping,
                     variants: Mapping, *, allow_missing_programs: bool = False) -> bool:
    """Tie every packaged Actor/report/program byte to its paired run report."""
    if (not isinstance(index, Mapping) or set(index) != set(p.VARIANTS)
            or not isinstance(variants, Mapping)
            or set(variants) != set(p.VARIANTS)):
        return False
    for variant in p.VARIANTS:
        bundle, trained = index[variant], variants[variant]
        if (not isinstance(bundle, Mapping)
                or set(bundle) != {"actor", "training_report", "programs"}
                or not isinstance(trained, Mapping)):
            return False
        actor_sha = _asset_bytes(root, release_id, bundle["actor"])[1]
        report, report_sha = _asset(root, release_id, bundle["training_report"])
        expected_report = {key: value for key, value in trained.items()
                           if key not in ("training_report_path",
                                          "training_report_sha256")}
        if (actor_sha != trained["actor_artifact"]["sha256"]
                or report_sha != trained["training_report_sha256"]
                or report != expected_report
                or trained["actor_artifact"].get("role_sha256") !=
                trained.get("selected_role_sha256")):
            return False
        programs = trained.get("program_artifacts") or {}
        required = set(p.ROLES) if variant == "regularized" else set()
        if (not isinstance(programs, Mapping)
                or set(programs) != set(bundle["programs"])
                or (set(programs) != required and not (
                    variant == "regularized" and allow_missing_programs
                    and not programs))):
            return False
        for role, item in programs.items():
            if (_asset_bytes(root, release_id,
                             bundle["programs"][role])[1] != item["sha256"]
                    or item.get("actor_role_sha256") !=
                    trained["selected_role_sha256"][role]):
                return False
    return True


def _paired_schedule_evidence(pair: Mapping) -> bool:
    """Verify the assigned block sequence and every actual environment reset.

    Early episode ends can give raw and regularized different reset lists. The
    immutable contract is the same seed assigned to each joint-step block,
    with both actual lists reported faithfully; equality is not required.
    """
    namespace = "warehouse-v6-matched-120-step-blocks"
    seed, steps = pair.get("pair_seed"), pair.get("joint_steps")
    if (type(seed) is not int or seed not in (*p.DEVELOPMENT_PAIR_SEEDS,
                                              *p.FORMAL_PAIR_SEEDS)
            or type(steps) is not int or steps not in p.BUDGETS
            or p.SCENARIO_BLOCK_STEPS != 120
            or not isinstance(pair.get("candidate_id"), str)
            or not isinstance(pair.get("release_id"), str)
            or not _sha(pair.get("source_closure_sha256"))):
        return False
    checkpoint_seeds = (p.DEVELOPMENT_SELECTION_SEEDS if
                        seed in p.DEVELOPMENT_PAIR_SEEDS else
                        p.FORMAL_SELECTION_SEEDS)
    expected_spec = p.PairRunSpec(
        pair["candidate_id"], seed, steps, "/ignored-by-scientific-hash",
        release_id=pair["release_id"], checkpoint_seeds=checkpoint_seeds)
    expected_spec_sha = expected_spec.sha256()
    expected_training_sha = expected_spec.training_config_sha256(
        pair["source_closure_sha256"])
    expected_schedule_sha = p.scenario_schedule_sha256(namespace, seed, steps)
    variants = pair.get("variants")
    if (not isinstance(variants, Mapping) or set(variants) != set(p.VARIANTS)
            or pair.get("run_spec_sha256") != expected_spec_sha
            or pair.get("training_config_sha256") != expected_training_sha
            or pair.get("architecture_mode") !=
            p.REGISTERED_ARCHITECTURE_MODE
            or pair.get("bc_recipe_version") != p.REGISTERED_BC_RECIPE_VERSION
            or pair.get("assigned_scenario_schedule_sha256") !=
            expected_schedule_sha
            or pair.get("online_program_extraction_steps") !=
            list(expected_spec.online_program_extraction_steps)
            or pair.get("checkpoint_seeds") != list(checkpoint_seeds)):
        return False
    actual = {}
    for variant in p.VARIANTS:
        report = variants[variant]
        if (not isinstance(report, Mapping)
                or report.get("variant") != variant
                or report.get("pair_seed") != seed
                or report.get("joint_steps") != steps
                or report.get("architecture_mode") !=
                p.REGISTERED_ARCHITECTURE_MODE
                or report.get("bc_recipe_version") !=
                p.REGISTERED_BC_RECIPE_VERSION
                or report.get("online_program_extraction_steps") !=
                list(expected_spec.online_program_extraction_steps)
                or report.get("online_program_split_block_cycle") !=
                p.FINAL_PROGRAM_SPLIT_COUNTS
                or report.get("online_program_cohorts_disjoint") is not True
                or report.get("run_spec_sha256") != expected_spec_sha
                or report.get("training_config_sha256") != expected_training_sha
                or report.get("scenario_schedule_sha256") !=
                expected_schedule_sha
                or report.get("assigned_scenario_schedule_kind") !=
                "fixed_120_joint_step_blocks"):
            return False
        resets = report.get("actual_episode_resets")
        if (not isinstance(resets, list) or not resets
                or report.get("actual_episode_resets_sha256") != _digest(resets)):
            return False
        cursor = 0
        for index, row in enumerate(resets):
            if not isinstance(row, Mapping) or set(row) != {
                    "seed", "episode_index", "episode_key", "start_joint_step",
                    "joint_steps", "complete", "shutdown_episode"}:
                return False
            episode_steps = row["joint_steps"]
            limit = min(p.SCENARIO_BLOCK_STEPS - cursor % p.SCENARIO_BLOCK_STEPS,
                        steps - cursor)
            if (type(row["seed"]) is not int
                    or type(row["episode_index"]) is not int
                    or type(row["start_joint_step"]) is not int
                    or type(episode_steps) is not int
                    or type(row["complete"]) is not bool
                    or type(row["shutdown_episode"]) is not bool
                    or row["episode_index"] != index
                    or row["start_joint_step"] != cursor
                    or row["seed"] != p.scenario_seed(namespace, seed, cursor)
                    or row["episode_key"] != f'{row["seed"]}:{index}'
                    or not 1 <= episode_steps <= limit
                    or (not row["complete"] and episode_steps != limit)):
                return False
            cursor += episode_steps
        if cursor != steps:
            return False
        actual[variant] = resets
    return (pair.get("actual_episode_reset_count") == {
        variant: len(actual[variant]) for variant in p.VARIANTS}
            and pair.get("actual_episode_resets_equal") is
            (actual["raw"] == actual["regularized"]))


def _summary(rows: Sequence[Mapping], expected_seeds: Sequence[int]) -> dict:
    if (not isinstance(rows, list) or len(rows) != len(expected_seeds)
            or [row.get("seed") for row in rows] != list(expected_seeds)):
        raise ValueError("Warehouse outcome rows or seeds changed")
    for row in rows:
        if (type(row.get("deliveries")) is not int or row["deliveries"] < 0
                or type(row.get("collisions")) is not int or row["collisions"] < 0
                or type(row.get("shutdown_episode")) is not bool):
            raise ValueError("Warehouse outcome has invalid count")
    size = len(rows)
    return {
        "mean_deliveries": sum(row["deliveries"] for row in rows) / size,
        "mean_collisions": sum(row["collisions"] for row in rows) / size,
        "shutdown_episode_rate": sum(row["shutdown_episode"] for row in rows) / size,
    }


def _same_summary(claimed: Mapping, measured: Mapping) -> bool:
    try:
        return all(math.isfinite(float(claimed[key]))
                   and abs(float(claimed[key]) - measured[key]) <= 1e-9
                   for key in measured)
    except (KeyError, TypeError, ValueError):
        return False


def _audit_metric(metric: Mapping, sample_count: int, *,
                  purpose: str, development: bool) -> bool:
    if not p.program_pass(metric, purpose=purpose, development=development):
        return False
    rows = metric.get("rows")
    actionable_rows = metric.get("actionable_rows")
    if (type(rows) is not int or rows > sample_count
            or type(actionable_rows) is not int
            or not 0 <= actionable_rows <= rows
            or metric.get("sample_count") != sample_count
            or not math.isclose(float(metric["coverage"]), rows / sample_count,
                                rel_tol=0, abs_tol=1e-9)
            or not math.isclose(float(metric["actionable_coverage"]),
                                actionable_rows / sample_count,
                                rel_tol=0, abs_tol=1e-9)):
        return False
    certified = metric.get("certified_rows")
    if certified is not None and (
            type(certified) is not int or not 0 <= certified <= rows):
        return False
    return True


def _program_audit(audits: Mapping, *, role_hashes: Mapping,
                   program_hashes: Mapping, expected_seeds: Sequence[int],
                   development: bool = False) -> bool:
    if not isinstance(audits, Mapping) or set(audits) != set(p.ROLES):
        return False
    for role in p.ROLES:
        row = audits[role]
        if (not isinstance(row, Mapping) or row.get("role") != role
                or row.get("actor_role_sha256") != role_hashes[role]
                or row.get("program_sha256") != program_hashes[role]
                or row.get("episode_ids") != sorted(map(str, expected_seeds))
                or row.get("episode_count") != len(expected_seeds)
                or type(row.get("sample_count")) is not int
                or row["sample_count"] < 1000
                or type(row.get("max_depth")) is not int or row["max_depth"] > 5
                or type(row.get("leaf_count")) is not int or row["leaf_count"] > 16
                or type(row.get("max_features_per_node")) is not int
                or row["max_features_per_node"] > 3
                or row.get("feedback_eligible") is not True
                or not _audit_metric(row.get("feedback", {}),
                                     row["sample_count"], purpose="feedback",
                                     development=development)):
            return False
        if (development or role == "robot_2") and (
                row.get("explanation_eligible") is not True or
                not _audit_metric(row.get("explanation", {}),
                                  row["sample_count"], purpose="explanation",
                                  development=development)):
            return False
    return True


def _audit_trace_index(audits: Mapping, index: Mapping, *, root: Path,
                       release_id: str) -> bool:
    """Bind every audit summary to its original content-addressed frame rows."""
    if not audits:
        return index == {}
    if (not isinstance(audits, Mapping) or set(audits) != set(p.ROLES)
            or not isinstance(index, Mapping) or set(index) != set(p.ROLES)):
        return False
    for role in p.ROLES:
        audit, item = audits[role], index[role]
        if (not isinstance(audit, Mapping) or not isinstance(item, Mapping)
                or set(item) != {"path", "sha256"}
                or not isinstance(audit.get("trace_path"), str)
                or not audit["trace_path"].endswith(".jsonl")
                or Path(audit["trace_path"]).name != Path(item["path"]).name
                or audit.get("trace_sha256") != item["sha256"]
                or type(audit.get("trace_row_count")) is not int
                or audit["trace_row_count"] != audit.get("sample_count")):
            return False
        raw, _ = _asset_bytes(root, release_id, item)
        if raw.count(b"\n") != audit["trace_row_count"] or not raw.endswith(b"\n"):
            return False
    return True


def _trace_actor_entry(assets: Mapping, trained: Mapping) -> dict:
    """Build the same role binding for a development pair as the release pair."""
    regularized = assets["regularized"]
    report = trained["regularized"]
    return {
        "actor_path": regularized["actor"]["path"],
        "actor_sha256": regularized["actor"]["sha256"],
        "role_sha256": {
            "human": report["selected_role_sha256"]["robot_1"],
            "ai": report["selected_role_sha256"]["robot_2"],
        },
        "program_paths": [regularized["programs"][role]["path"]
                          for role in p.ROLES],
        "program_sha256": {role: regularized["programs"][role]["sha256"]
                           for role in p.ROLES},
    }


def _same_recomputed(claimed: object, measured: object) -> bool:
    """Compare all independently derived fields; only trace metadata may be extra."""
    if isinstance(measured, Mapping):
        return (isinstance(claimed, Mapping)
                and all(key in claimed and _same_recomputed(claimed[key], value)
                        for key, value in measured.items()))
    if isinstance(measured, (list, tuple)):
        return (isinstance(claimed, (list, tuple))
                and len(claimed) == len(measured)
                and all(_same_recomputed(left, right)
                        for left, right in zip(claimed, measured, strict=True)))
    if isinstance(measured, bool) or measured is None or isinstance(measured, str):
        return type(claimed) is type(measured) and claimed == measured
    if isinstance(measured, int):
        return type(claimed) is int and claimed == measured
    # Different deployment CPUs can vary the final bit of a NumPy reduction.
    # Discrete actions, leaves, episode identity and every asset hash remain exact.
    return (type(claimed) in (int, float)
            and math.isfinite(float(claimed))
            and math.isclose(float(claimed), float(measured),
                             rel_tol=0.0, abs_tol=1e-6))


def _replay_environment(seed: int):
    """Recreate the exact neutral NN–NN study state without trainer imports."""
    from env.warehouse.domain import participant_study_config
    from env.warehouse.environment import WarehouseMultiAgentEnv

    environment = WarehouseMultiAgentEnv(participant_study_config())
    environment.reset(seed=seed)
    # The trainer's state restoration also refreshes public navigation state.
    # A direct field assignment would not reproduce the recorded pre-state.
    marked = environment.get_state()
    marked.participant_controlled_agent_id = "robot_1"
    environment.set_state(marked)
    return environment


def _public_frame_for_trace(state, previous_actions: Mapping | None) -> dict:
    """Project a replay state onto the fields available in a v6 public frame.

    The history fold below must not read AgentState's cached action, battery,
    or position summaries.  The only previous decisions come from the prior
    physical transition, just as they do in an exported participant frame.
    """
    from env.warehouse.layouts import STUDY_MAP_LAYOUT
    from env.warehouse.public_observation_v6 import FEATURE_SCHEMA_SHA256

    roles = {"human": "robot_1", "ai": "robot_2"}
    frame = {
        "turn": int(state.frame),
        "episode_id": int(state.episode_id),
        "map_id": STUDY_MAP_LAYOUT.layout_id,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "previous_joint_actions": previous_actions,
        "previous_collision_kind": (state.last_robot_collision_kind
                                    if state.frame else None),
        "same_target_collision_streak": int(state.same_target_collision_streak),
        "orders": [{
            "id": task.task_id,
            "pickup": [task.pickup_position[1], task.pickup_position[0]],
            "dropoff": [task.delivery_position[1], task.delivery_position[0]],
            "status": task.status,
            "carrier": next((role for role, agent_id in roles.items()
                             if agent_id == task.carrier_agent_id), None),
        } for task in state.tasks],
    }
    for role, agent_id in roles.items():
        agent = state.by_id(agent_id)
        frame[role] = {
            "x": int(agent.position[1]), "y": int(agent.position[0]),
            "battery": float(agent.battery),
            "carrying": agent.carrying_task_id,
            "active": bool(agent.active), "heading": agent.heading.lower(),
        }
    return frame


def _verify_program_trace_rows(*, actor, programs: Mapping,
                               trace_bytes_by_role: Mapping[str, bytes],
                               audits: Mapping,
                               expected_seeds: Sequence[int]) -> bool:
    """Recompute a saved audit from its exact local or packaged trace bytes.

    This replays the recorded physical trajectory for integrity, but never
    launches a new NN rollout or spends an independent evaluation seed.
    """
    import numpy as np

    from backend.training.warehouse_v6_program import audit_frozen_program
    from env.warehouse.decision_protocol import (
        canonical_sha256, independent_agent_seed)
    from env.warehouse.navigation import ACTIONS
    from env.warehouse.public_observation_v6 import (
        OBSERVATION_DIM, action_masks, encode_policy_features,
        encode_public_observation, public_pre_action_snapshot,
        rebuild_public_pre_action_state,
        rebuild_public_pre_action_state_from_frames)

    if (set(programs) != set(p.ROLES)
            or set(trace_bytes_by_role) != set(p.ROLES)
            or set(audits) != set(p.ROLES)):
        return False
    identities, records = {}, {}
    for role in p.ROLES:
        program = programs[role]
        raw = trace_bytes_by_role[role]
        digest = sha256(raw).hexdigest()
        audit = audits[role]
        if (program.role != role
                or program.actor_role_sha256 != actor.role_sha256[role]
                or audit.get("trace_sha256") != digest
                or audit.get("trace_row_count") != audit.get("sample_count")):
            return False
        samples, ordered, recorded = [], [], []
        for line in raw.splitlines():
            row = json.loads(line)
            if not isinstance(row, Mapping):
                return False
            base_seed, episode_id, frame = (row.get("base_seed"),
                                            row.get("episode_id"), row.get("frame"))
            if (type(base_seed) is not int or base_seed not in expected_seeds
                    or type(episode_id) is not int or episode_id < 0
                    or type(frame) is not int or frame < 0
                    or row.get("role") != role
                    or row.get("actor_sha256") != actor.artifact_sha256
                    or row.get("actor_role_sha256") != actor.role_sha256[role]
                    or row.get("program_sha256") != program.sha256()):
                return False
            seed = independent_agent_seed(
                base_seed=base_seed, episode_id=episode_id,
                frame=frame, agent_id=role)
            if (row.get("rng_seed") != seed
                    or row.get("pre_state_public_sha256") !=
                    canonical_sha256(row.get("pre_state_public"))):
                return False
            obs = np.asarray(row.get("obs"), dtype=np.float32)
            mask = np.asarray(row.get("mask"), dtype=np.float32)
            if (obs.shape != (OBSERVATION_DIM,)
                    or mask.shape != (len(ACTIONS),)
                    or not np.isfinite(obs).all()
                    or not np.isfinite(mask).all()
                    or np.any((mask != 0.0) & (mask != 1.0))
                    or mask.sum() < 1):
                return False
            encoded = encode_policy_features(obs, mask)
            logits = actor.roles[role].logits_encoded(encoded[None])[0]
            masked = np.where(mask > .5, logits, -1.0e9)
            probabilities = np.exp(masked - masked.max())
            probabilities /= probabilities.sum()
            action = ACTIONS[int(np.random.default_rng(seed).choice(
                len(ACTIONS), p=probabilities))]
            top = ACTIONS[int(probabilities.argmax())]
            claimed_probabilities = np.asarray(
                row.get("nn_probabilities"), dtype=np.float64)
            if (claimed_probabilities.shape != (len(ACTIONS),)
                    or not np.isfinite(claimed_probabilities).all()
                    or not np.allclose(claimed_probabilities, probabilities,
                                       rtol=0.0, atol=1e-7)
                    or row.get("sampled_action") != action
                    or row.get("nn_sampled_action") != action
                    or row.get("nn_argmax_action") != top
                    or mask[ACTIONS.index(action)] <= .5):
                return False
            public = row["pre_state_public"]
            if (not isinstance(public, Mapping)
                    or public.get("frame") != frame
                    or public.get("episode_id") != episode_id):
                return False
            rebuilt = rebuild_public_pre_action_state(dict(public))
            role_index = p.ROLES.index(role)
            if (not np.array_equal(
                    obs, encode_public_observation(rebuilt)[role_index])
                    or not np.array_equal(
                        mask, action_masks(rebuilt)[role_index])):
                return False
            post = row.get("post_state_public")
            if (not isinstance(post, Mapping)
                    or row.get("post_state_public_sha256") !=
                    canonical_sha256(post)):
                return False
            samples.append({"episode": str(base_seed), "obs": obs,
                            "mask": mask, "sampled_action": action})
            ordered.append((base_seed, episode_id, frame))
            recorded.append(row)
        if len(samples) != audit["sample_count"] or len(set(ordered)) != len(ordered):
            return False
        if [seed for seed in expected_seeds if any(row[0] == seed for row in ordered)] != list(
                expected_seeds):
            return False
        if ordered != sorted(ordered):
            return False
        measured = audit_frozen_program(program, actor, samples)
        if not _same_recomputed(audit, measured):
            return False
        identities[role] = ordered
        records[role] = recorded
    if identities["robot_1"] != identities["robot_2"]:
        return False
    # Integrity replay authenticates the saved action-before-state chain. It
    # never selects a checkpoint or computes a fresh quality estimate.
    cursor = 0
    for seed in expected_seeds:
        env = _replay_environment(int(seed))
        public_frames = []
        previous_actions = None
        while not (env.state.terminated or env.state.truncated):
            if cursor >= len(identities["robot_1"]):
                return False
            key = (int(seed), int(env.state.episode_id), int(env.state.frame))
            if identities["robot_1"][cursor] != key:
                return False
            pre = public_pre_action_snapshot(env.state)
            obs, masks = encode_public_observation(env.state), action_masks(env.state)
            public_frames.append(_public_frame_for_trace(
                env.state, previous_actions))
            from_frames = rebuild_public_pre_action_state_from_frames(public_frames)
            if (not np.array_equal(obs, encode_public_observation(from_frames))
                    or not np.array_equal(masks, action_masks(from_frames))):
                return False
            actions, distributions = actor.act(env.state, base_seed=int(seed))
            for index_role, role in enumerate(p.ROLES):
                row = records[role][cursor]
                if (row["pre_state_public"] != pre
                        or not np.array_equal(np.asarray(row["obs"], dtype=np.float32),
                                              obs[index_role])
                        or not np.array_equal(np.asarray(row["mask"], dtype=np.float32),
                                              masks[index_role])
                        or row["nn_sampled_action"] != actions[role]
                        or not np.allclose(row["nn_probabilities"],
                                           distributions[role].probabilities,
                                           rtol=0.0, atol=1e-7)):
                    return False
            _, _, _, _, info = env.step(actions)
            if (info["proposed_actions"] != actions
                    or info["action_override_agents"]):
                return False
            previous_actions = {
                ui_role: {
                    "proposed": actions[agent_id].lower(),
                    "executed": info["executed_actions"][agent_id].lower(),
                }
                for ui_role, agent_id in (("human", "robot_1"),
                                          ("ai", "robot_2"))
            }
            post = public_pre_action_snapshot(env.state)
            if any(records[role][cursor]["post_state_public"] != post
                   for role in p.ROLES):
                return False
            cursor += 1
    return cursor == len(identities["robot_1"])


def _verify_program_trace(*, root: Path, release_id: str,
                          index: Mapping, audits: Mapping,
                          entry: Mapping,
                          expected_seeds: Sequence[int]) -> bool:
    """Verify content-addressed release traces against both frozen Actors.

    The shared row verifier also serves admission before one-use seed claims.
    This wrapper preserves release path and asset hash requirements.
    """
    from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor
    from backend.training.warehouse_v6_program import WarehouseV6Program

    actor = FrozenWarehouseV6Actor.load(
        root / entry["actor_path"], expected_sha256=entry["actor_sha256"],
        expected_role_sha256={
            "robot_1": entry["role_sha256"]["human"],
            "robot_2": entry["role_sha256"]["ai"],
        })
    if set(index) != set(p.ROLES) or set(audits) != set(p.ROLES):
        return False
    programs, trace_bytes_by_role = {}, {}
    for role, program_path in zip(p.ROLES, entry["program_paths"], strict=True):
        programs[role] = WarehouseV6Program.load(
            root / program_path,
            expected_sha256=entry["program_sha256"][role],
            actor_role_sha256=actor.role_sha256[role])
        raw, digest = _asset_bytes(root, release_id, index[role])
        if (audits[role].get("trace_sha256") != digest
                or audits[role].get("trace_row_count") !=
                audits[role].get("sample_count")):
            return False
        trace_bytes_by_role[role] = raw
    return _verify_program_trace_rows(
        actor=actor, programs=programs,
        trace_bytes_by_role=trace_bytes_by_role,
        audits=audits, expected_seeds=expected_seeds)


def _final_fit_verified_rows(*, root: Path, release_id: str,
                             index: Mapping, pair: Mapping,
                             assets: Mapping) -> dict:
    """Replay all four post-selection fit splits against the frozen dual NN.

    The returned rows are suitable for an independent offline `fit_program`
    replay at packaging time. Runtime calls this function too, but never needs
    Torch or the trainer. A trace row is accepted only if the real environment
    produces its pre/post state and both seeded NN actions.
    """
    import numpy as np

    from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor
    from env.warehouse.decision_protocol import (
        canonical_sha256, independent_agent_seed)
    from env.warehouse.navigation import ACTIONS
    from env.warehouse.public_observation_v6 import (
        action_masks, encode_public_observation, public_pre_action_snapshot)

    seed = pair["pair_seed"]
    schedule = p.final_program_fit_seed_schedule(release_id, seed)
    trained = pair["variants"]["regularized"]
    bundle = assets["regularized"]
    reports = trained["final_fresh_program_reports"]
    if (set(index) != set(schedule) or set(reports) != set(p.ROLES)
            or pair.get("release_id") != release_id
            or trained.get("release_id") != release_id
            or trained.get("final_program_fit_seed_schedule") != schedule
            or trained.get("final_program_fit_seed_schedule_sha256") != _digest(schedule)
            or trained["actor_artifact"]["sha256"] != bundle["actor"]["sha256"]):
        raise ValueError("Warehouse final fit schedule or Actor differs")
    actor = FrozenWarehouseV6Actor.load(
        root / bundle["actor"]["path"],
        expected_sha256=bundle["actor"]["sha256"],
        expected_role_sha256=trained["selected_role_sha256"])
    used_training = {row["seed"] for row in trained["training_episodes"]}
    if used_training & {value for seeds in schedule.values() for value in seeds}:
        raise ValueError("Warehouse final fit reused a training episode")
    traces = reports[p.ROLES[0]]["final_fit_traces"]
    if (set(traces) != set(schedule)
            or any(reports[role].get("final_fit_traces") != traces
                   or reports[role].get("split_seeds") != schedule
                   or reports[role].get("split_seed_schedule_sha256") != _digest(schedule)
                   or reports[role].get("actor_artifact_sha256") != actor.artifact_sha256
                   or reports[role].get("release_id") != release_id
                   for role in p.ROLES)):
        raise ValueError("Warehouse final fit report trace inventory differs")
    samples = {role: {} for role in p.ROLES}
    for split, seeds in schedule.items():
        claim, asset = traces[split], index[split]
        if (set(claim) != {"path", "sha256", "row_count", "seeds"}
                or set(asset) != {"path", "sha256"}
                or claim["seeds"] != seeds
                or claim["sha256"] != asset["sha256"]
                or Path(claim["path"]).name != Path(asset["path"]).name
                or Path(asset["path"]).name != f"{split}.jsonl"):
            raise ValueError("Warehouse final fit trace identity differs")
        raw, _ = _asset_bytes(root, release_id, asset)
        if (not raw.endswith(b"\n") or raw.count(b"\n") != claim["row_count"]
                or claim["row_count"] < len(seeds)):
            raise ValueError("Warehouse final fit trace row count differs")
        saved = iter(raw.splitlines())
        for scenario_seed in seeds:
            env = _replay_environment(int(scenario_seed))
            while not (env.state.terminated or env.state.truncated):
                try:
                    row = json.loads(next(saved))
                except StopIteration as exc:
                    raise ValueError("Warehouse final fit frame missing") from exc
                if (not isinstance(row, Mapping)
                        or set(row) != {"base_seed", "episode_id", "frame",
                                           "pre_state_public", "pre_state_public_sha256",
                                           "post_state_public", "post_state_public_sha256",
                                           "actor_sha256", "roles"}
                        or any(type(row[name]) is not int for name in
                               ("base_seed", "episode_id", "frame"))
                        or (row["base_seed"], row["episode_id"], row["frame"]) !=
                        (scenario_seed, env.state.episode_id, env.state.frame)
                        or row["actor_sha256"] != actor.artifact_sha256
                        or set(row["roles"]) != set(p.ROLES)):
                    raise ValueError("Warehouse final fit frame identity differs")
                pre = public_pre_action_snapshot(env.state)
                if (row["pre_state_public"] != pre
                        or row["pre_state_public_sha256"] != canonical_sha256(pre)):
                    raise ValueError("Warehouse final fit pre-state differs")
                obs, masks = encode_public_observation(env.state), action_masks(env.state)
                actions, distributions = actor.act(env.state, base_seed=int(scenario_seed))
                for position, role in enumerate(p.ROLES):
                    role_row = row["roles"][role]
                    if (not isinstance(role_row, Mapping)
                            or set(role_row) != {"obs", "mask", "sampled_action",
                                                 "nn_probabilities", "rng_seed",
                                                 "actor_role_sha256"}):
                        raise ValueError("Warehouse final fit role row incomplete")
                    probabilities = np.asarray(
                        role_row["nn_probabilities"], dtype=np.float64)
                    role_obs = np.asarray(role_row["obs"], dtype=np.float32)
                    role_mask = np.asarray(role_row["mask"], dtype=np.float32)
                    expected_rng = independent_agent_seed(
                        base_seed=int(scenario_seed),
                        episode_id=int(env.state.episode_id),
                        frame=int(env.state.frame), agent_id=role)
                    if (not np.array_equal(role_obs, obs[position])
                            or not np.array_equal(role_mask, masks[position])
                            or probabilities.shape != (len(ACTIONS),)
                            or not np.isfinite(probabilities).all()
                            or not np.allclose(probabilities,
                                               distributions[role].probabilities,
                                               rtol=0.0, atol=1e-7)
                            or role_row["sampled_action"] != actions[role]
                            or type(role_row["rng_seed"]) is not int
                            or role_row["rng_seed"] != expected_rng
                            or role_row["actor_role_sha256"] !=
                            actor.role_sha256[role]
                            or role_mask[ACTIONS.index(actions[role])] <= .5):
                        raise ValueError("Warehouse final fit NN action differs")
                    samples[role].setdefault(split, []).append({
                        "episode": str(scenario_seed),
                        "base_seed": int(scenario_seed),
                        "obs": role_obs, "mask": role_mask,
                        "sampled_action": actions[role],
                    })
                _, _, _, _, info = env.step(actions)
                if (info["proposed_actions"] != actions
                        or info["action_override_agents"]):
                    raise ValueError("Warehouse final fit action was rewritten")
                post = public_pre_action_snapshot(env.state)
                if (row["post_state_public"] != post
                        or row["post_state_public_sha256"] != canonical_sha256(post)):
                    raise ValueError("Warehouse final fit post-state differs")
        if next(saved, None) is not None:
            raise ValueError("Warehouse final fit contains an extra frame")
    return samples


def _final_fit_index_claim(*, root: Path, release_id: str,
                           index: Mapping, pair: Mapping,
                           assets: Mapping) -> bool:
    """Check unselected split bytes and schedules against offline replay receipts.

    Packaging physically replays and refits every pair. At runtime the selected
    formal pair is replayed again; other pairs use their content hashes, fixed
    seed schedule and package-time refit receipt so a cold study process does
    not replay hundreds of thousands of development frames.
    """
    schedule = p.final_program_fit_seed_schedule(release_id, pair["pair_seed"])
    trained = pair["variants"]["regularized"]
    reports = trained["final_fresh_program_reports"]
    actor_sha = trained["actor_artifact"]["sha256"]
    if (pair.get("release_id") != release_id
            or trained.get("release_id") != release_id
            or trained.get("final_program_fit_seed_schedule") != schedule
            or trained.get("final_program_fit_seed_schedule_sha256") != _digest(schedule)
            or assets["regularized"]["actor"]["sha256"] != actor_sha
            or set(reports) != set(p.ROLES)
            or set(index) != set(schedule)):
        return False
    traces = reports[p.ROLES[0]].get("final_fit_traces")
    if not isinstance(traces, Mapping) or set(traces) != set(schedule):
        return False
    programs = trained.get("program_artifacts") or {}
    for role in p.ROLES:
        claim = reports[role]
        if (claim.get("final_fit_traces") != traces
                or claim.get("actor_artifact_sha256") != actor_sha
                or claim.get("split_seeds") != schedule
                or claim.get("split_seed_schedule_sha256") != _digest(schedule)
                or claim.get("release_id") != release_id):
            return False
        if claim.get("status") == "frozen_actor_fresh_four_part_fit":
            if (role not in programs
                    or claim.get("actor_role_sha256") !=
                    trained["selected_role_sha256"][role]
                    or claim.get("program_sha256") != programs[role]["sha256"]
                    or claim.get("validation") != trained.get(
                        "selected_program_reports", {}).get(role, {}).get("validation")):
                return False
        elif claim.get("status") != "failed" or role in programs:
            return False
    for split, seeds in schedule.items():
        claim, item = traces[split], index[split]
        if (not isinstance(claim, Mapping)
                or set(claim) != {"path", "sha256", "row_count", "seeds"}
                or not isinstance(item, Mapping)
                or set(item) != {"path", "sha256"}
                or claim["seeds"] != seeds
                or claim["sha256"] != item["sha256"]
                or Path(claim["path"]).name != f"{split}.jsonl"
                or Path(item["path"]).name != f"{split}.jsonl"
                or type(claim["row_count"]) is not int
                or claim["row_count"] < len(seeds)):
            return False
        raw, _ = _asset_bytes(root, release_id, item)
        if raw.count(b"\n") != claim["row_count"] or not raw.endswith(b"\n"):
            return False
    return True


def _verify_final_fit_pair(*, root: Path, release_id: str,
                           index: Mapping, pair: Mapping,
                           assets: Mapping,
                           verified_samples: Mapping | None = None) -> bool:
    """Derive final leaf qualification and validation from independent rows."""
    from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor
    from backend.training.warehouse_v6_program import (
        WarehouseV6Program, _eligible, _metrics, _passes,
        _sample_arrays, actor_sparse_topology_matches,
        ACTOR_SPARSE_TOPOLOGY_METHOD, FEEDBACK_AGREEMENT,
        EXPLANATION_AGREEMENT, MAX_COVERED_KL)

    samples = verified_samples if verified_samples is not None else \
        _final_fit_verified_rows(
            root=root, release_id=release_id, index=index, pair=pair, assets=assets)
    trained = pair["variants"]["regularized"]
    reports = trained["final_fresh_program_reports"]
    programs = trained.get("program_artifacts") or {}
    if set(programs) != set(assets["regularized"]["programs"]):
        return False
    actor = FrozenWarehouseV6Actor.load(
        root / assets["regularized"]["actor"]["path"],
        expected_sha256=assets["regularized"]["actor"]["sha256"],
        expected_role_sha256=trained["selected_role_sha256"])
    for role in p.ROLES:
        claim = reports[role]
        if claim.get("status") == "failed":
            if role in programs or role in trained.get("selected_program_reports", {}):
                return False
            continue
        if (claim.get("status") != "frozen_actor_fresh_four_part_fit"
                or role not in programs
                or claim.get("actor_role_sha256") != actor.role_sha256[role]
                or claim.get("program_sha256") != programs[role]["sha256"]):
            return False
        program = WarehouseV6Program.load(
            root / assets["regularized"]["programs"][role]["path"],
            expected_sha256=programs[role]["sha256"],
            actor_role_sha256=actor.role_sha256[role])
        report = program.report
        if (trained["selected_program_reports"].get(role) != report
                or claim.get("validation") != report.get("validation")):
            return False
        fit_supported = None
        if report.get("fit_method") == ACTOR_SPARSE_TOPOLOGY_METHOD:
            if not actor_sparse_topology_matches(program, actor.roles[role]):
                return False
            fit_x, _ = _sample_arrays(samples[role]["fit"], "final program fit")
            fit_leaves = program.leaf_indices(fit_x)
            counts = np.bincount(fit_leaves, minlength=len(program.leaf_logits))
            if (not np.array_equal(fit_leaves, actor.roles[role].leaf(fit_x))
                    or report.get("fit_leaf_row_counts") !=
                        counts.astype(int).tolist()):
                return False
            fit_supported = set(np.flatnonzero(counts).tolist())
        elif report.get("fit_method") is not None:
            return False
        expected_ids = {name: sorted(map(str, seeds)) for name, seeds in
                        p.final_program_fit_seed_schedule(
                            release_id, pair["pair_seed"]).items()}
        if (report.get("split_episode_ids") != expected_ids
                or report.get("split_episodes") != {
                    name: len(seeds) for name, seeds in expected_ids.items()}
                or report.get("split_episode_sha256") != {
                    name: sha256(json.dumps(seeds, separators=(",", ":")).encode()
                                 ).hexdigest() for name, seeds in expected_ids.items()}):
            return False
        leaf_reports = report.get("leaf_eligibility")
        if not isinstance(leaf_reports, Mapping) or set(leaf_reports) != set(
                map(str, range(program.leaf_logits.shape[0]))):
            return False
        selected = {"feedback": set(), "explanation": set()}
        for leaf_text, metrics in leaf_reports.items():
            leaf = int(leaf_text)
            for purpose, threshold in (("feedback", FEEDBACK_AGREEMENT),
                                       ("explanation", EXPLANATION_AGREEMENT)):
                measured = _metrics(program, actor.roles[role],
                                    samples[role]["eligibility"], {leaf}, purpose)
                if not _same_recomputed(metrics.get(purpose), measured):
                    return False
                if ((fit_supported is None or leaf in fit_supported)
                        and _eligible(measured, threshold)):
                    selected[purpose].add(leaf)
        if (selected["feedback"] != set(program.feedback_leaves)
                or selected["explanation"] != set(program.explanation_leaves)):
            return False
        validation = {}
        for purpose, threshold in (("feedback", FEEDBACK_AGREEMENT),
                                   ("explanation", EXPLANATION_AGREEMENT)):
            validation[purpose] = _metrics(
                program, actor.roles[role], samples[role]["validation"],
                selected[purpose], purpose)
            measured = validation[purpose]
            validation[purpose + "_online_passed"] = bool(
                measured["rows"] > 0 and measured["agreement"] >= threshold
                and measured["kl"] is not None
                and measured["kl"] <= MAX_COVERED_KL
                and measured["illegal_count"] == 0)
            validation[purpose + "_passed"] = _passes(measured, threshold)
            validation["development_" + purpose + "_passed"] = _passes(
                measured, threshold, development=True)
        if not _same_recomputed(report.get("validation"), validation):
            return False
    return True


def _final_fit_replay_receipt(*, root: Path, release_id: str,
                              item: Mapping, pair: Mapping,
                              pair_sha256: str, index: Mapping,
                              source_closure_sha256: str) -> bool:
    """Bind the package-time Torch refit result to exact immutable inputs."""
    replay, _ = _asset(root, release_id, item)
    trained = pair["variants"]["regularized"]
    expected = {
        "format": "warehouse-final-fit-replay.v1",
        "release_id": release_id,
        "candidate_id": pair["candidate_id"],
        "pair_seed": pair["pair_seed"],
        "joint_steps": pair["joint_steps"],
        "source_closure_sha256": source_closure_sha256,
        "pair_report_sha256": pair_sha256,
        "actor_sha256": trained["actor_artifact"]["sha256"],
        "trace_sha256": {name: index[name]["sha256"]
                         for name in p.FINAL_PROGRAM_SPLIT_COUNTS},
        "refitted_program_sha256": {
            role: (trained.get("program_artifacts") or {}).get(role, {}).get("sha256")
            for role in p.ROLES},
        "passed": True,
    }
    return replay == expected


def _stress_summary(stress: Mapping) -> dict | None:
    """Derive the quality summary from every registered pressure trajectory."""
    schedule = p.stress_schedule()
    rows = stress.get("episodes")
    if (not isinstance(rows, list) or len(rows) != len(schedule)
            or stress.get("schedule_sha256") != _digest(schedule)):
        return None
    for row, (task, seed, action_seed) in zip(rows, schedule, strict=True):
        if (not isinstance(row, Mapping)
                or (row.get("task"), row.get("scenario_seed"),
                    row.get("action_seed")) != (task, seed, action_seed)
                or type(row.get("deliveries")) is not int
                or row["deliveries"] < 0
                or type(row.get("collisions")) is not int
                or row["collisions"] < 0
                or type(row.get("shutdown_episode")) is not bool):
            return None
    n = len(rows)
    measured = {
        "mean_deliveries": sum(row["deliveries"] for row in rows) / n,
        "mean_collisions": sum(row["collisions"] for row in rows) / n,
        "shutdown_episode_rate": sum(row["shutdown_episode"] for row in rows) / n,
    }
    return measured if _same_summary(stress.get("summary", {}), measured) else None


def _quality_replay_receipt(*, root: Path, release_id: str, item: Mapping,
                            actor_sha256: str, role_sha256: Mapping,
                            rows: Sequence[Mapping],
                            seeds: Sequence[tuple[int, int]]) -> bool:
    """Bind an offline full NN/physics replay to its exact quality rows.

    Packaging physically replays every saved frame once. Runtime rechecks the
    content-addressed certificate against all original trace bytes and the
    packaged Actor identity without repeating sealed statistical evaluation.
    """
    from .quality_replay import claimed_series

    replay, _ = _asset(root, release_id, item)
    return replay == claimed_series(
        actor_sha256=actor_sha256, role_sha256=role_sha256,
        rows=rows, seeds=seeds)


def _development_pass(row: Mapping) -> bool:
    """Recompute all quality, training, split audit, and provenance gates."""
    training = row.get("training") or {}
    if (not p.development_pair_pass(row)
            or row.get("program_audit_seeds") !=
            list(p.DEVELOPMENT_PROGRAM_AUDIT_SEEDS)
            or set(row.get("stress", {})) != set(p.VARIANTS)
            or row.get("stress_episodes") != len(p.stress_schedule())):
        return False
    for variant in p.VARIANTS:
        measured = _stress_summary(row["stress"][variant])
        if (measured is None or
                row["stress"][variant].get("actor_sha256") !=
                training[variant]["actor_artifact"]["sha256"] or
                not _same_summary(row["evaluation"][variant], measured)):
            return False
    regularized = training["regularized"]
    role_hashes = regularized["selected_role_sha256"]
    artifacts = regularized.get("program_artifacts") or {}
    if set(artifacts) != set(p.ROLES):
        return False
    program_hashes = {}
    for role in p.ROLES:
        if artifacts[role].get("actor_role_sha256") != role_hashes[role]:
            return False
        program_hashes[role] = artifacts[role]["sha256"]
    return _program_audit(
        row["program_audit"], role_hashes=role_hashes,
        program_hashes=program_hashes,
        expected_seeds=p.DEVELOPMENT_PROGRAM_AUDIT_SEEDS, development=True)


def _heldout(outcomes: Mapping, release_id: str) -> bool:
    if not isinstance(outcomes, Mapping) or set(outcomes) != set(p.VARIANTS):
        return False
    schedule = p.heldout_schedule(release_id)
    for variant in p.VARIANTS:
        result = outcomes[variant]
        tasks = result.get("tasks")
        if (result.get("episode_count") != 300 or not isinstance(tasks, Mapping)
                or set(tasks) != {"1", "2", "3"}):
            return False
        all_rows = []
        for task in p.TASKS:
            section = tasks[str(task)]
            rows = section["episodes"]
            measured = _summary(rows, schedule[task])
            if (not _same_summary(section["summary"], measured)
                    or section.get("episodes_sha256") != _digest(rows)):
                return False
            all_rows.extend(rows)
        pooled = _summary(all_rows, tuple(seed for task in p.TASKS
                                          for seed in schedule[task]))
        if (not _same_summary(result["summary"], pooled)
                or not p.quality_pass(pooled, development=False)):
            return False
    return True


def validate_acceptance(acceptance: Mapping, *, release_id: str,
                        entries: Mapping, evidence: Mapping,
                        repository_root: str | Path) -> bool:
    """Validate v6 acceptance and its complete content-addressed run package.

    `entries` is the two-variant Warehouse catalog. `evidence` is the result
    of `release_catalog.warehouse_evidence()`. Any absent or altered link keeps
    the pilot closed. The function has no recovery or fallback policy path.
    """
    try:
        return _validate(acceptance, release_id=release_id, entries=entries,
                         evidence=evidence, root=Path(repository_root))
    except (KeyError, IndexError, TypeError, ValueError, OSError, StopIteration,
            OverflowError):
        return False


def _validate(receipt: Mapping, *, release_id: str, entries: Mapping,
              evidence: Mapping, root: Path) -> bool:
    if (not isinstance(receipt, Mapping)
            or receipt.get("release") != release_id
            or receipt.get("release_id") != release_id
            or receipt.get("domain") != "warehouse"
            or receipt.get("passed") is not True
            or receipt.get("heldout_evaluated") is not True
            or receipt.get("post_policy_interventions") != 0
            or receipt.get("post_policy_override_count") != 0
            or receipt.get("all_candidate_regularized_feedback_passed") is not True
            or receipt.get("locked_budget") not in p.BUDGETS
            or receipt.get("pair_seeds") != list(p.FORMAL_PAIR_SEEDS)
            or receipt.get("selected_pair_seed") not in p.FORMAL_PAIR_SEEDS
            or receipt.get("policy_id") != _policy_ids(release_id)
            or receipt.get("joint_steps") != {variant: receipt["locked_budget"]
                                               for variant in p.VARIANTS}
            or not _sha(receipt.get("source_closure_sha256"))
            or set(entries) != set(p.VARIANTS)):
        return False
    budget = receipt["locked_budget"]
    for variant in p.VARIANTS:
        entry = entries[variant]
        expected_roles = {"human": receipt["role_sha256"][variant]["robot_1"],
                          "ai": receipt["role_sha256"][variant]["robot_2"]}
        if (entry.get("policy_id") != receipt["policy_id"][variant]
                or entry.get("actor_sha256") != receipt["artifact_sha256"][variant]
                or entry.get("training_report_sha256") !=
                receipt["training_report_sha256"][variant]
                or entry.get("role_sha256") != expected_roles
                or (variant == "regularized" and
                    entry.get("program_sha256") != receipt["program_sha256"][variant])):
            return False
    phase_index = evidence["phase_receipts"]
    claim_index = evidence["one_use_claims"]
    pair_index = evidence["pair_reports"]
    training_asset_index = evidence["training_assets"]
    trace_index = evidence["program_traces"]
    fit_trace_index = evidence["final_fit_traces"]
    fit_replay_index = evidence["final_fit_replays"]
    quality_replay_index = evidence["quality_replays"]
    if (set(phase_index) != set(_PHASE_NAMES)
            or set(claim_index) != {"final_program_audit", "heldout"}
            or set(pair_index) != set(map(str, p.FORMAL_PAIR_SEEDS))
            or set(training_asset_index) != set(pair_index)
            or set(trace_index) != {"development", "preaudit", "final_audit"}
            or set(fit_trace_index) != {"development", "formal"}
            or set(fit_replay_index) != {"development", "formal"}
            or set(fit_trace_index["formal"]) != set(pair_index)
            or set(fit_replay_index["formal"]) != set(pair_index)
            or set(quality_replay_index) != {"development", "formal", "heldout"}
            or set(quality_replay_index["formal"]) != set(pair_index)):
        return False
    phases = {}
    for name in _PHASE_NAMES:
        row, file_sha = _asset(root, release_id, phase_index[name])
        if (receipt["phase_receipt_sha256"].get(name) != file_sha
                or row.get("release_id") != release_id
                or row.get("candidate_id") != receipt["candidate_id"]):
            return False
        phases[name] = row
    lock = phases["budget.lock.json"]
    selected = phases["formal.selection.json"]
    program_schedules = {
        str(seed): p.final_program_fit_seed_schedule(release_id, seed)
        for seed in (*p.DEVELOPMENT_PAIR_SEEDS, *p.FORMAL_PAIR_SEEDS)}
    if (lock.get("passed") is not True
            or lock.get("locked_budget") != budget
            or lock.get("source_closure_sha256") != receipt["source_closure_sha256"]
            or lock.get("architecture_mode") !=
            p.REGISTERED_ARCHITECTURE_MODE
            or lock.get("bc_recipe_version") !=
            p.REGISTERED_BC_RECIPE_VERSION
            or lock.get("online_program_extraction_steps") !=
            list(p.online_program_extraction_steps(budget))
            or lock.get("final_program_fit_seed_schedule") != program_schedules
            or lock.get("final_program_fit_seed_schedule_sha256") !=
            _digest(program_schedules)
            or lock.get("development_pair_seeds") != list(p.DEVELOPMENT_PAIR_SEEDS)
            or selected.get("passed") is not True
            or selected.get("locked_budget") != budget
            or selected.get("selected_pair_seed") != receipt["selected_pair_seed"]
            or selected.get("pair_seeds") != list(p.FORMAL_PAIR_SEEDS)
            or selected.get("selection_seeds") != list(p.FORMAL_SELECTION_SEEDS)
            or selected.get("selection_rule") !=
            "maximin_deliveries_minus_0.25_collisions_minus_5_shutdown"):
        return False
    # A smaller registered budget cannot be skipped. Recompute its three
    # independent development gates from the packaged immutable receipts.
    dev_index = evidence["development_receipts"]
    dev_pair_index = evidence["development_pair_reports"]
    dev_asset_index = evidence["development_training_assets"]
    expected_budgets = p.BUDGETS[:p.BUDGETS.index(budget) + 1]
    if (set(dev_index) != set(map(str, expected_budgets))
            or set(dev_pair_index) != set(dev_index)
            or set(dev_asset_index) != set(dev_index)
            or set(trace_index["development"]) != set(dev_index)
            or set(fit_trace_index["development"]) != set(dev_index)
            or set(fit_replay_index["development"]) != set(dev_index)
            or set(quality_replay_index["development"]) != set(dev_index)):
        return False
    dev_reports = {}
    for candidate_budget in expected_budgets:
        section = dev_index[str(candidate_budget)]
        pairs_section = dev_pair_index[str(candidate_budget)]
        assets_section = dev_asset_index[str(candidate_budget)]
        traces_section = trace_index["development"][str(candidate_budget)]
        fit_traces_section = fit_trace_index["development"][str(candidate_budget)]
        fit_replays_section = fit_replay_index["development"][str(candidate_budget)]
        quality_replays_section = quality_replay_index["development"][str(candidate_budget)]
        if (set(section) != set(map(str, p.DEVELOPMENT_PAIR_SEEDS))
                or set(pairs_section) != set(section)
                or set(assets_section) != set(section)
                or set(traces_section) != set(section)
                or set(fit_traces_section) != set(section)
                or set(fit_replays_section) != set(section)
                or set(quality_replays_section) != set(section)):
            return False
        dev_reports[candidate_budget] = []
        for seed in p.DEVELOPMENT_PAIR_SEEDS:
            row, file_sha = _asset(root, release_id, section[str(seed)])
            pair, pair_sha = _asset(root, release_id, pairs_section[str(seed)])
            passed = _development_pass(row)
            if (row.get("release_id") != release_id
                    or row.get("candidate_id") != receipt["candidate_id"]
                    or row.get("pair_seed") != seed
                    or row.get("budget") != candidate_budget
                    or row.get("source_closure_sha256") !=
                    receipt["source_closure_sha256"]
                    or row.get("stress_schedule_sha256") !=
                    _digest(p.stress_schedule())
                    or row.get("pair_report_sha256") != pair_sha
                    or row.get("training") != pair.get("variants")
                    or pair.get("candidate_id") != receipt["candidate_id"]
                    or pair.get("pair_seed") != seed
                    or pair.get("release_id") != release_id
                    or pair.get("final_program_fit_seed_schedule") !=
                    program_schedules[str(seed)]
                    or pair.get("final_program_fit_seed_schedule_sha256") !=
                    _digest(program_schedules[str(seed)])
                    or pair.get("joint_steps") != candidate_budget
                    or pair.get("source_closure_sha256") !=
                    receipt["source_closure_sha256"]
                    or not _paired_schedule_evidence(pair)
                    or not _training_assets(
                        root, release_id, assets_section[str(seed)],
                        pair["variants"],
                        allow_missing_programs=candidate_budget < budget)
                    or not _final_fit_index_claim(
                        root=root, release_id=release_id,
                        index=fit_traces_section[str(seed)], pair=pair,
                        assets=assets_section[str(seed)])
                    or not _final_fit_replay_receipt(
                        root=root, release_id=release_id,
                        item=fit_replays_section[str(seed)], pair=pair,
                        pair_sha256=pair_sha,
                        index=fit_traces_section[str(seed)],
                        source_closure_sha256=receipt["source_closure_sha256"])
                    or not _audit_trace_index(
                        row.get("program_audit", {}), traces_section[str(seed)],
                        root=root, release_id=release_id)
                    or (bool(row.get("program_audit")) and
                        not _verify_program_trace(
                            root=root, release_id=release_id,
                            index=traces_section[str(seed)],
                            audits=row["program_audit"],
                            entry=_trace_actor_entry(
                                assets_section[str(seed)], pair["variants"]),
                            expected_seeds=p.DEVELOPMENT_PROGRAM_AUDIT_SEEDS))
                    or row.get("passed") is not passed):
                return False
            if row.get("stress") and (
                    set(row["stress"]) != set(p.VARIANTS)
                    or any(_stress_summary(row["stress"][variant]) is None
                           for variant in p.VARIANTS)):
                return False
            quality_items = quality_replays_section[str(seed)]
            if set(quality_items) != (set(p.VARIANTS) if row.get("stress") else set()):
                return False
            for variant in quality_items:
                if not _quality_replay_receipt(
                        root=root, release_id=release_id,
                        item=quality_items[variant],
                        actor_sha256=pair["variants"][variant]["actor_artifact"]["sha256"],
                        role_sha256=pair["variants"][variant]["selected_role_sha256"],
                        rows=row["stress"][variant]["episodes"],
                        seeds=[(scenario_seed, action_seed)
                               for _, scenario_seed, action_seed in p.stress_schedule()]):
                    return False
            if candidate_budget == budget and lock["development_receipt_sha256"].get(
                    str(seed)) != file_sha:
                return False
            dev_reports[candidate_budget].append(row)
    first_passing = next((candidate_budget for candidate_budget in expected_budgets
                          if all(_development_pass(row)
                                 for row in dev_reports[candidate_budget])), None)
    if first_passing != budget:
        return False
    formal_rows, pair_reports = [], {}
    candidates = receipt["candidate_training"]
    if (not isinstance(candidates, list) or len(candidates) != 3
            or [row.get("pair_seed") for row in candidates] !=
            list(p.FORMAL_PAIR_SEEDS)):
        return False
    for candidate in candidates:
        seed = candidate["pair_seed"]
        pair, pair_file_sha = _asset(root, release_id, pair_index[str(seed)])
        formal = phases[f"formal.training.{seed}.json"]
        if (candidate.get("pair_report_sha256") != pair_file_sha
                or formal.get("pair_report_sha256") != pair_file_sha
                or pair.get("pair_seed") != seed
                or pair.get("release_id") != release_id
                or pair.get("final_program_fit_seed_schedule") !=
                program_schedules[str(seed)]
                or pair.get("final_program_fit_seed_schedule_sha256") !=
                _digest(program_schedules[str(seed)])
                or pair.get("joint_steps") != budget
                or pair.get("source_closure_sha256") != receipt["source_closure_sha256"]
                or not _paired_schedule_evidence(pair)
                or pair.get("initial_role_sha256") != candidate.get("initial_role_sha256")
                or not p.paired_training_pass(pair.get("variants", {}), joint_steps=budget)
                or not _training_assets(root, release_id,
                                        training_asset_index[str(seed)],
                                        pair["variants"])
                or not (
                    _verify_final_fit_pair if seed == receipt["selected_pair_seed"]
                    else _final_fit_index_claim)(
                        root=root, release_id=release_id,
                        index=fit_trace_index["formal"][str(seed)], pair=pair,
                        assets=training_asset_index[str(seed)])
                or not _final_fit_replay_receipt(
                    root=root, release_id=release_id,
                    item=fit_replay_index["formal"][str(seed)], pair=pair,
                    pair_sha256=pair_file_sha,
                    index=fit_trace_index["formal"][str(seed)],
                    source_closure_sha256=receipt["source_closure_sha256"])
                or pair.get("checkpoint_seeds") != list(p.FORMAL_SELECTION_SEEDS)
                or formal.get("passed") is not all(
                    p.quality_pass(formal["variants"][variant]["quality"],
                                   development=False) for variant in p.VARIANTS)
                or formal.get("selection_seeds") != list(p.FORMAL_SELECTION_SEEDS)
                or formal.get("pair_seed") != seed
                or selected["pair_receipt_sha256"].get(str(seed)) !=
                receipt["phase_receipt_sha256"][f"formal.training.{seed}.json"]):
            return False
        for variant in p.VARIANTS:
            report = pair["variants"][variant]
            summary = formal["variants"][variant]
            if (not _same_summary(summary["quality"], _summary(
                    summary["episodes"], p.FORMAL_SELECTION_SEEDS))
                    or summary.get("actor_sha256") != report["actor_artifact"]["sha256"]
                    or summary.get("training_report_sha256") !=
                    report["training_report_sha256"]
                    or candidate["variants"][variant]["selected_role_sha256"] !=
                    report["selected_role_sha256"]
                    or candidate["variants"][variant]["initial_role_sha256"] !=
                    report["initial_role_sha256"]):
                return False
            if (set(quality_replay_index["formal"][str(seed)]) != set(p.VARIANTS)
                    or not _quality_replay_receipt(
                        root=root, release_id=release_id,
                        item=quality_replay_index["formal"][str(seed)][variant],
                        actor_sha256=report["actor_artifact"]["sha256"],
                        role_sha256=report["selected_role_sha256"],
                        rows=summary["episodes"],
                        seeds=[(item, item) for item in p.FORMAL_SELECTION_SEEDS])):
                return False
        formal_rows.append(formal)
        pair_reports[seed] = pair
    if p.select_formal_pair(formal_rows)["pair_seed"] != receipt["selected_pair_seed"]:
        return False
    selected_pair = pair_reports[receipt["selected_pair_seed"]]
    for variant in p.VARIANTS:
        trained = selected_pair["variants"][variant]
        if (trained["actor_artifact"]["sha256"] !=
                receipt["artifact_sha256"][variant]
                or trained["selected_role_sha256"] != receipt["role_sha256"][variant]
                or trained["training_report_sha256"] !=
                receipt["training_report_sha256"][variant]
                or trained["feedback_rows"] != receipt[
                    "raw_program_feedback_rows" if variant == "raw" else
                    "regularized_program_feedback_rows"]):
            return False
    programs = receipt["program_sha256"]["regularized"]
    if set(programs) != set(p.ROLES):
        return False
    for phase, expected_seeds in (
            ("program.preaudit.json", p.FORMAL_PROGRAM_PREAUDIT_SEEDS),
            ("final_program_audit.json", p.final_audit_seeds(release_id))):
        row = phases[phase]
        trace_name = ("preaudit" if phase == "program.preaudit.json"
                      else "final_audit")
        if (row.get("passed") is not True
                or row.get("selected_pair_seed") != receipt["selected_pair_seed"]
                or row.get("actor_sha256") != receipt["artifact_sha256"]["regularized"]
                or row.get("program_sha256") != programs
                or row.get("seeds") != list(expected_seeds)
                or not _audit_trace_index(
                    row["audit"], trace_index[trace_name], root=root,
                    release_id=release_id)
                or not _verify_program_trace(
                    root=root, release_id=release_id,
                    index=trace_index[trace_name], audits=row["audit"],
                    entry=entries["regularized"],
                    expected_seeds=expected_seeds)
                or not _program_audit(row["audit"],
                                      role_hashes=receipt["role_sha256"]["regularized"],
                                      program_hashes=programs,
                                      expected_seeds=expected_seeds)):
            return False
    if receipt["final_program_audit"] != phases["final_program_audit.json"]["audit"]:
        return False
    held = phases["heldout.json"]
    if (held.get("passed") is not True
            or held.get("selected_pair_seed") != receipt["selected_pair_seed"]
            or held.get("seed_rule") != "base_seed_times_10_plus_task"
            or held.get("base_seeds") != list(p.heldout_base_seeds(release_id))
            or held.get("outcomes") != receipt["heldout"]
            or receipt.get("heldout_scenario_seed_rule") != held["seed_rule"]
            or receipt.get("heldout_seed_ranges") != {str(task): [
                p.heldout_base_seeds(release_id)[0],
                p.heldout_base_seeds(release_id)[-1]] for task in p.TASKS}
            or not _heldout(receipt["heldout"], release_id)):
        return False
    if set(quality_replay_index["heldout"]) != set(p.VARIANTS):
        return False
    for variant in p.VARIANTS:
        if held["outcomes"][variant].get("actor_sha256") != \
                selected_pair["variants"][variant]["actor_artifact"]["sha256"]:
            return False
        if set(quality_replay_index["heldout"][variant]) != set(map(str, p.TASKS)):
            return False
        for task in p.TASKS:
            if not _quality_replay_receipt(
                    root=root, release_id=release_id,
                    item=quality_replay_index["heldout"][variant][str(task)],
                    actor_sha256=selected_pair["variants"][variant][
                        "actor_artifact"]["sha256"],
                    role_sha256=selected_pair["variants"][variant][
                        "selected_role_sha256"],
                    rows=held["outcomes"][variant]["tasks"][str(task)]["episodes"],
                    seeds=[(seed, seed) for seed in
                           p.heldout_schedule(release_id)[task]]):
                return False
    for phase in ("final_program_audit", "heldout"):
        claim, claim_sha = _asset(root, release_id, claim_index[phase])
        expected_seeds = (p.final_audit_seeds(release_id) if phase ==
                          "final_program_audit" else p.heldout_base_seeds(release_id))
        if (receipt["one_use_claim_sha256"].get(phase) != claim_sha
                or phases[f"{phase}.json"].get("one_use_claim_sha256") != claim_sha
                or claim.get("candidate_id") != receipt["candidate_id"]
                or claim.get("release_id") != release_id
                or claim.get("phase") != phase
                or claim.get("seeds") != list(expected_seeds)
                or claim.get("seed_sha256") != _digest(list(expected_seeds))
                or claim.get("actor_hashes") != receipt["artifact_sha256"]):
            return False
    return True
