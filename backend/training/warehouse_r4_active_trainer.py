"""R4 continuation that trains a more active warehouse neural teammate.

The continuation starts from the frozen 3.95M Alignment Actor and keeps all
five Actor logits authoritative.  Program partners only contribute ordinary
environment trajectories.  The periodically fitted RCPD program contributes
an explicitly bounded KL(pi_nn || pi_program) term during training and is
never queried to choose, repair, or replace a runtime action.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from core.program import ExecutableProgram
from env.warehouse.navigation import ACTIONS, MOVE_DELTAS, shortest_path_distance
from env.warehouse.layouts import get_map_layout
from env.warehouse.domain import collaborative_study_config
from env.warehouse_native.policy import NumPyNativeActor
from env.warehouse_native.partners import partner_action
from env.warehouse_native.feedback import FeedbackConfig, FeedbackManager
from .warehouse_native_public_feedback import PublicFeedbackEnvironment
from .warehouse_native_public_feedback_evaluation import REWARD as PUBLIC_FEEDBACK_REWARD
from env.warehouse_native.scenarios import reset_scenario
from . import warehouse_family_alignment_run as source_run
from . import warehouse_family_alignment_trainer as source_trainer
from . import warehouse_r4_role_feedback_update as feedback_update
from .warehouse_native import gae
from .warehouse_native_common import digest, file_hash
from .warehouse_native_partner_mix_run import decode
from .warehouse_native_partner_mix_trainer import _prefix
from .warehouse_native_public_feedback_initialization import initialization_sha256
from .warehouse_native_public_feedback_trainer import (
    COUNTERS, EPISODE_FIELDS, PublicFeedbackTrainer, _cpu,
)
from .warehouse_native_v2 import NativeV2Trainer


VERSION = "warehouse-r4-active-trainer.v18"
SOURCE_ROOT = Path("output/warehouse_native/alignment_50k_pair_20260910")
SOURCE_ACTOR_SHA256 = "309b6e53fe682bead8d3443015aca27eae60e561175e71d7c25f57314ac69d5b"
SOURCE_CHECKPOINT_SHA256 = "c76b77385e2003896ed6d598d2cd8897d42beebe4be62ac4090943d4204b1ec5"
SOURCE_CUMULATIVE_STEPS = 3_950_000
PARTNER_MIX = {"selfplay": .20, "skilled": .20, "assertive": .50, "noisy": .10}
SHAPING = {
    "productive_progress": .02,
    "avoidable_wait": -.03,
    "task_distance_regression": -.02,
}
KL_MAX = .01
TRAINING_KL_LAMBDA = .001
ACTOR_LR = 1e-4
CRITIC_LR = 1e-4
ENTROPY_COEFFICIENT = .001
ORDINARY_EPISODE_PROBABILITY = .85
HARD_STATE_EPISODE_PROBABILITY = .15
COLLISION_RECOVERY_EPISODE_PROBABILITY = 0.0
LOW_ENERGY_EPISODE_PROBABILITY = 0.0
LOW_ENERGY_ROBOT_2_BATTERY_MAX = 70.0
CHARGE_CURRICULUM_VERSION = "warehouse-r4-public-charge-trajectory-curriculum.v2"
CHARGE_CURRICULUM_SOURCE_SCENARIOS = 128
CHARGE_CURRICULUM_SOURCE_FRAMES = 32
CHARGE_CURRICULUM_MAX_ROWS = 4096
RECOVERY_CURRICULUM_VERSION = "warehouse-r4-v8-program-partner-recovery-curriculum.v2"
RECOVERY_CURRICULUM_SOURCE_SCENARIOS = 128
RECOVERY_CURRICULUM_SOURCE_PARTNERS = ("skilled", "assertive", "noisy")
RECOVERY_CURRICULUM_PRE_SHUTDOWN_WINDOW = 20
PROGRAM_PARTNER_PROBABILITY = 1.0 - PARTNER_MIX["selfplay"]
CONDITIONAL_COLLISION_RECOVERY_PROBABILITY = (
    COLLISION_RECOVERY_EPISODE_PROBABILITY / PROGRAM_PARTNER_PROBABILITY)
CONDITIONAL_LOW_ENERGY_PROBABILITY = (
    LOW_ENERGY_EPISODE_PROBABILITY / PROGRAM_PARTNER_PROBABILITY)
CONDITIONAL_HARD_STATE_PROBABILITY = (
    HARD_STATE_EPISODE_PROBABILITY / PROGRAM_PARTNER_PROBABILITY)
HARD_STATE_CURRICULUM_VERSION = "warehouse-r4-v8-public-hard-state-curriculum.v1"
HARD_STATE_MAX_PER_PARTNER_GROUP = 64
HARD_STATE_NO_PROGRESS_THRESHOLD = 18
V8_STAGE_ROOT = Path("output/warehouse_native/r4_active_lowentropy_v8_1m_20260911")
V8_STAGE_ACTOR_SHA256 = "bc9c269738d31565460065747d855dfb026fa2a841432cd612149de24a7cce5c"
V8_STAGE_CHECKPOINT_SHA256 = "ea86439b36171946a4491b7eafd78737e16b7a08073cc9d4108da2aea4f61145"
V8_STAGE_ADDITIONAL_STEPS = 50_000
V12_STAGE_ACTOR_SHA256 = "bfef0a2fc4f5744949f092a2ac894bb52a7f895050de13063c56114ac64d2460"
V12_STAGE_CHECKPOINT_SHA256 = "64daaf65fc7834d7fab9ed56f64b9f031aca5104065d5dda8dcadcb8e3fb1a95"
V12_STAGE_ADDITIONAL_STEPS = 100_000
TRAINING_PROGRAM_MIN_FIDELITY = .85
TRAINING_PROGRAM_MIN_CRITICAL_FIDELITY = .80
FINAL_EXPLANATION_MIN_FIDELITY = .90
FINAL_EXPLANATION_MIN_CRITICAL_FIDELITY = .85
TREE_CONFIG = FeedbackConfig(
    warmup_steps=0,
    ramp_steps=0,
    lambda_max=KL_MAX,
    # The portable explainer accepts at most depth12/256 leaves.  Select the
    # simplest member that reaches the registered fidelity gates.
    depths=(4, 6, 8, 10, 12),
    leaves=(16, 32, 64, 128, 256),
    min_samples_leaf=4,
    minimum_fidelity=TRAINING_PROGRAM_MIN_FIDELITY,
    minimum_critical_fidelity=TRAINING_PROGRAM_MIN_CRITICAL_FIDELITY,
    minimum_training_rows=256,
    minimum_validation_rows=128,
)
RESERVOIR_ROWS = 12_000


def _state_dict_sha256(state):
    return sha256(json.dumps(
        state, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _unknown_public_history(state):
    agent_ids = [agent["agent_id"] for agent in state["agents"]]
    return {
        "valid": False,
        "frame": int(state["frame"]),
        "previous_frame": None,
        "submitted_actions": None,
        "executed_actions": None,
        "move_canceled": None,
        "collision_kind": None,
        "post_state_sha256": _state_dict_sha256(state),
        "unknown_reason": "no_verified_previous_transition",
        "consecutive_move_canceled": {agent_id: 0 for agent_id in agent_ids},
        "consecutive_collision": 0,
        "previous_counts": None,
    }


def _derive_charge_margin_entry(entry):
    """Create one legal public-state curriculum snapshot from a train entry.

    The only changed state field is robot_2's visible battery.  The derived
    value pays for a shortest path to the public charger plus the configured
    safety margin, while remaining below the public mission energy budget.
    """
    source_snapshot = entry["snapshot"]
    source_snapshot_sha256 = digest(source_snapshot)
    snapshot = deepcopy(source_snapshot)
    config, state = snapshot["configuration"], snapshot["state"]
    agent = state["agents"][1]
    from env.warehouse.layouts import get_map_layout
    charger = get_map_layout(config["map_layout_id"]).charger_position
    charger_distance = shortest_path_distance(
        tuple(agent["position"]), tuple(charger), config["map_layout_id"]
    )
    if not np.isfinite(charger_distance):
        return None
    original_battery = float(agent["battery"])
    required_to_charger = float(config["move_battery_cost"]) * charger_distance
    derived_battery = min(
        original_battery,
        required_to_charger + float(config["battery_safety_margin"]),
    )
    if derived_battery <= 0 or derived_battery >= original_battery:
        return None
    agent["battery"] = float(derived_battery)
    source_history = deepcopy(source_snapshot.get("public_feedback_history"))
    if source_history is not None:
        # This is integrity metadata, not an additional public state change.
        # A counterfactual battery cannot claim the original previous-step
        # history hash, so the new episode starts with explicitly unknown history.
        snapshot["public_feedback_history"] = _unknown_public_history(state)
    # Prove that no field other than the public battery value differs.
    restored = deepcopy(snapshot)
    restored["state"]["agents"][1]["battery"] = original_battery
    if source_history is not None:
        restored["public_feedback_history"] = source_history
    if restored != source_snapshot or digest(source_snapshot) != source_snapshot_sha256:
        raise RuntimeError("Charge curriculum attempted to mutate a source snapshot")
    _, charge_needed, _ = _goal_distance(snapshot, 1)
    if not charge_needed or derived_battery < required_to_charger:
        return None
    recipe = {
        "version": CHARGE_CURRICULUM_VERSION,
        "source_entry_id": entry["id"],
        "source_fingerprint": entry["fingerprint"],
        "source_snapshot_sha256": source_snapshot_sha256,
        "changed_public_field": "state.agents[1].battery",
        "rebound_integrity_metadata": (["public_feedback_history"]
                                       if source_history is not None else []),
        "original_battery": original_battery,
        "derived_battery": float(derived_battery),
        "charger_distance": int(charger_distance),
        "move_battery_cost": float(config["move_battery_cost"]),
        "battery_safety_margin": float(config["battery_safety_margin"]),
        "initial_charge_needed": True,
        "reachable_without_shutdown": True,
    }
    return {
        "id": f"{entry['id']}__charge_margin",
        "snapshot": snapshot,
        "derived_snapshot_sha256": digest(snapshot),
        "recipe": recipe,
        "recipe_sha256": digest(recipe),
    }


def _build_charge_margin_curriculum(entries):
    source_sha256 = digest(entries)
    derived = tuple(filter(None, (_derive_charge_margin_entry(entry) for entry in entries)))
    if not derived or digest(entries) != source_sha256:
        raise ValueError("No immutable train snapshots support the charge curriculum")
    manifest = {
        "version": CHARGE_CURRICULUM_VERSION,
        "source_split": "train",
        "source_entries_sha256": source_sha256,
        "derived_count": len(derived),
        "derived": [{
            "id": row["id"],
            "derived_snapshot_sha256": row["derived_snapshot_sha256"],
            "recipe_sha256": row["recipe_sha256"],
        } for row in derived],
    }
    manifest["sha256"] = digest(manifest)
    return derived, manifest


def _build_trajectory_charge_margin_curriculum(
        entries, actor_path, *, maximum_scenarios=CHARGE_CURRICULUM_SOURCE_SCENARIOS,
        maximum_frames=CHARGE_CURRICULUM_SOURCE_FRAMES,
        maximum_rows=CHARGE_CURRICULUM_MAX_ROWS, environment=None):
    """Generate varied train-only source states, then change only public battery."""
    actor_path = Path(actor_path).resolve()
    if file_hash(actor_path) != SOURCE_ACTOR_SHA256:
        raise ValueError("Charge curriculum must use the exact frozen r3 Actor")
    source_entries_sha256 = digest(entries)
    actor = NumPyNativeActor(actor_path)
    env = environment or PublicFeedbackEnvironment(
        collaborative_study_config(), PUBLIC_FEEDBACK_REWARD,
        collision_cost=.05, mode="observed",
    )
    rows, seen, positions = [], set(), set()
    distance_bins = Counter()
    excluded = Counter()
    evidence_steps = action_count = 0
    used_scenarios = 0
    for entry in entries[:maximum_scenarios]:
        reset_scenario(env, entry)
        used_scenarios += 1
        for _ in range(maximum_frames):
            source_snapshot = env.snapshot()
            source = {
                "id": f"{entry['id']}__trajectory_frame_{env.state.frame:03d}",
                "fingerprint": env.fingerprint(),
                "snapshot": source_snapshot,
            }
            row = _derive_charge_margin_entry(source)
            if row is not None:
                inter_robot_distance = shortest_path_distance(
                    tuple(source_snapshot["state"]["agents"][0]["position"]),
                    tuple(source_snapshot["state"]["agents"][1]["position"]),
                    source_snapshot["configuration"]["map_layout_id"],
                )
                if row["recipe"]["charger_distance"] == 0:
                    excluded["already_at_charger"] += 1
                    row = None
                elif inter_robot_distance <= 2:
                    excluded["robots_within_two_path_steps"] += 1
                    row = None
            if row is not None and row["derived_snapshot_sha256"] not in seen:
                row["recipe"].update({
                    "source_kind": "frozen_r3_deterministic_train_trajectory",
                    "source_scene_id": entry["id"],
                    "source_frame": int(env.state.frame),
                })
                row["recipe_sha256"] = digest(row["recipe"])
                rows.append(row)
                seen.add(row["derived_snapshot_sha256"])
                positions.add(tuple(source_snapshot["state"]["agents"][1]["position"]))
                distance_bins[str(row["recipe"]["charger_distance"])] += 1
                if len(rows) >= maximum_rows:
                    break
            actions, probabilities = actor.act(env.observations(), deterministic=True)
            for agent_id in env.agent_ids:
                if actions[agent_id] != ACTIONS[int(np.argmax(probabilities[agent_id]))]:
                    raise RuntimeError("Curriculum source Actor action is not its argmax")
            _, _, terminated, truncated, info = env.step(actions)
            evidence_steps += 1
            action_count += len(actions)
            if info["requested_actions"] != actions:
                raise RuntimeError("Curriculum source Actor command was overwritten")
            if terminated or truncated:
                break
        if len(rows) >= maximum_rows:
            break
    if not rows or digest(entries) != source_entries_sha256:
        raise ValueError("Train trajectory curriculum changed or produced no legal states")
    manifest = {
        "version": CHARGE_CURRICULUM_VERSION,
        "source_split": "train",
        "validation_and_test_excluded": True,
        "excluded_splits": ["calibration", "validation", "extraction",
                            "explanation_test", "final_test", "play"],
        "source_actor_sha256": SOURCE_ACTOR_SHA256,
        "source_entries_sha256": source_entries_sha256,
        "source_scenarios": used_scenarios,
        "maximum_source_frames_per_scenario": maximum_frames,
        "evidence_environment_steps": evidence_steps,
        "ppo_joint_steps": 0,
        "derived_count": len(rows),
        "robot_2_position_coverage": [list(position) for position in sorted(positions)],
        "robot_2_position_coverage_count": len(positions),
        "charger_distance_counts": dict(sorted(distance_bins.items(), key=lambda item: int(item[0]))),
        "minimum_inter_robot_path_distance": 3,
        "excluded_state_counts": dict(excluded),
        "source_neural_actions": action_count,
        "source_action_overrides": 0,
        "derived": [{
            "id": row["id"],
            "derived_snapshot_sha256": row["derived_snapshot_sha256"],
            "recipe_sha256": row["recipe_sha256"],
        } for row in rows],
    }
    manifest["sha256"] = digest(manifest)
    return tuple(rows), manifest


def _exact_curriculum_row(snapshot, *, entry, partner, bucket, source_frame,
                          future_shutdown_within=None, collision_kind=None):
    """Bind one immutable, unlabeled public trajectory state to its provenance."""
    frozen = deepcopy(snapshot)
    snapshot_sha256 = digest(frozen)
    recipe = {
        "version": RECOVERY_CURRICULUM_VERSION,
        "source_split": "train",
        "source_scene_id": entry["id"],
        "source_scene_fingerprint": entry["fingerprint"],
        "source_partner": partner,
        "source_frame": int(source_frame),
        "bucket": bucket,
        "source_snapshot_sha256": snapshot_sha256,
        "snapshot_mutation": False,
        "action_label_stored": False,
    }
    if future_shutdown_within is not None:
        recipe["future_robot_2_shutdown_within"] = int(future_shutdown_within)
    if collision_kind is not None:
        recipe["collision_kind"] = str(collision_kind)
    return {
        "id": f"{entry['id']}__{partner}__frame_{source_frame:03d}__{bucket}",
        "snapshot": frozen,
        "source_snapshot_sha256": snapshot_sha256,
        "recipe": recipe,
        "recipe_sha256": digest(recipe),
    }


def _public_reference_recoverability(row, *, maximum_steps):
    """Admission-only recovery check; its actions are never stored or trained on."""
    snapshot = row["snapshot"]
    frozen_sha256 = row["source_snapshot_sha256"]
    partner = row["recipe"]["source_partner"]
    env = PublicFeedbackEnvironment(
        collaborative_study_config(), PUBLIC_FEEDBACK_REWARD,
        collision_cost=.05, mode="observed",
    )
    env.restore(snapshot, require_feedback=True)
    rng_seed = int(sha256(row["id"].encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(rng_seed)
    charger = get_map_layout(env.config.map_layout_id).charger_position
    reached_charger = tuple(env.state.agents[1].position) == tuple(charger)
    productive_within = None
    task_progress_within = None
    longest_collision = collision_streak = 0
    action_count = 0
    for step in range(1, maximum_steps + 1):
        if env.done:
            break
        before = env.snapshot()
        before_sha256 = digest(before)
        # The registered source partner remains robot_1.  A public skilled
        # reference checks feasibility for robot_2 without supplying a label.
        player_action = partner_action(env, "robot_1", partner, rng)
        if digest(env.snapshot()) != before_sha256:
            raise RuntimeError("Recoverability partner mutated the source state")
        reference_action = partner_action(env, "robot_2", "skilled", rng)
        if digest(env.snapshot()) != before_sha256:
            raise RuntimeError("Recoverability reference mutated the source state")
        submitted = {"robot_1": player_action, "robot_2": reference_action}
        _, _, terminated, truncated, info = env.step(submitted)
        action_count += 2
        after = env.snapshot()
        _, flags = shaping_for_transition({
            "before": before, "after": after,
            "requested_actions": submitted,
            "executed_actions": info["executed_actions"],
            "events": info["events"],
        }, 1)
        if flags["productive"] and productive_within is None:
            productive_within = step
        if (task_progress_within is None and any(
                event.get("event") in ("pickup", "delivery")
                for event in info["events"])):
            task_progress_within = step
        collision_streak = collision_streak + 1 if info["robot_collision"] else 0
        longest_collision = max(longest_collision, collision_streak)
        reached_charger = reached_charger or (
            tuple(env.state.agents[1].position) == tuple(charger))
        if terminated or truncated:
            break
        if reached_charger and productive_within is not None:
            break
    if digest(snapshot) != frozen_sha256:
        raise RuntimeError("Recoverability check mutated its exact source snapshot")
    return {
        "robot_2_active": bool(env.state.agents[1].active),
        "reached_charger": bool(reached_charger),
        "first_productive_step": productive_within,
        "first_task_progress_step": task_progress_within,
        "longest_collision_streak": int(longest_collision),
        "reference_environment_steps": int(step),
        "reference_actions": int(action_count),
        "action_labels_stored": False,
    }


def _build_v8_program_partner_recovery_curriculum(
        entries, actor_path, *, maximum_scenarios=RECOVERY_CURRICULUM_SOURCE_SCENARIOS,
        maximum_frames=None, environment=None):
    """Collect exact v8 states after collisions and before observed shutdowns.

    Program partners act before Actor inference, cannot inspect its output and
    never mutate the environment.  Rows preserve the real public-history state
    byte-for-byte and contain no desired-action label.
    """
    actor_path = Path(actor_path).resolve()
    if file_hash(actor_path) != V8_STAGE_ACTOR_SHA256:
        raise ValueError("Recovery curriculum must use the exact frozen v8 Actor")
    source_entries_sha256 = digest(entries)
    actor = NumPyNativeActor(actor_path)
    env = environment or PublicFeedbackEnvironment(
        collaborative_study_config(), PUBLIC_FEEDBACK_REWARD,
        collision_cost=.05, mode="observed",
    )
    horizon = int(maximum_frames or collaborative_study_config().horizon)
    collision_rows, energy_rows = [], []
    collision_seen, energy_seen = set(), set()
    collision_kinds, source_partner_counts = Counter(), Counter()
    evidence_steps = source_neural_actions = 0
    shutdown_episodes = 0
    used_scenarios = 0
    for scene_index, entry in enumerate(entries[:maximum_scenarios]):
        used_scenarios += 1
        for partner_index, partner in enumerate(RECOVERY_CURRICULUM_SOURCE_PARTNERS):
            reset_scenario(env, entry)
            rng = np.random.default_rng(260_911_700 + scene_index * 101 + partner_index)
            prior_collision = False
            recoverable_energy = []
            for _ in range(horizon):
                before = env.snapshot()
                before_sha256 = digest(before)
                agent = before["state"]["agents"][1]
                charge_distance, charge_needed, _ = _goal_distance(before, 1)
                move_cost = float(before["configuration"]["move_battery_cost"])
                if (agent["active"] and charge_needed and charge_distance > 0
                        and float(agent["battery"]) >= move_cost * charge_distance):
                    recoverable_energy.append(_exact_curriculum_row(
                        before, entry=entry, partner=partner,
                        bucket="recoverable_charge_needed",
                        source_frame=before["state"]["frame"],
                    ))
                # This order is part of the information-boundary contract.
                player_action = partner_action(env, "robot_1", partner, rng)
                if player_action not in ACTIONS or digest(env.snapshot()) != before_sha256:
                    raise RuntimeError("Program partner mutated the recovery source state")
                actions, probabilities = actor.act(env.observations(), deterministic=True)
                for agent_id in env.agent_ids:
                    distribution = np.asarray(probabilities[agent_id], dtype=np.float64)
                    if actions[agent_id] != ACTIONS[int(np.argmax(distribution))]:
                        raise RuntimeError("Recovery source action differs from v8 argmax")
                submitted = {"robot_1": player_action, "robot_2": actions["robot_2"]}
                _, _, terminated, truncated, info = env.step(submitted)
                evidence_steps += 1
                source_neural_actions += 1
                if info["requested_actions"] != submitted:
                    raise RuntimeError("Recovery source neural action was overwritten")
                collision = bool(info["robot_collision"])
                if collision and not prior_collision and not (terminated or truncated):
                    after = env.snapshot()
                    history = after.get("public_feedback_history") or {}
                    if (not history.get("valid") or not history.get("collision_kind")
                            or history.get("post_state_sha256")
                                != _state_dict_sha256(after["state"])):
                        raise RuntimeError("Collision recovery row lacks verified public history")
                    row = _exact_curriculum_row(
                        after, entry=entry, partner=partner,
                        bucket="first_collision_recovery",
                        source_frame=after["state"]["frame"],
                        collision_kind=history["collision_kind"],
                    )
                    if row["source_snapshot_sha256"] not in collision_seen:
                        collision_rows.append(row)
                        collision_seen.add(row["source_snapshot_sha256"])
                        collision_kinds[history["collision_kind"]] += 1
                        source_partner_counts[f"collision.{partner}"] += 1
                prior_collision = collision
                if terminated or truncated:
                    break
            if not env.state.agents[1].active:
                shutdown_episodes += 1
                selected = recoverable_energy[-RECOVERY_CURRICULUM_PRE_SHUTDOWN_WINDOW:]
                total = len(selected)
                for index, candidate in enumerate(selected):
                    candidate = deepcopy(candidate)
                    candidate["recipe"]["bucket"] = "pre_shutdown_recoverable"
                    candidate["recipe"]["future_robot_2_shutdown_within"] = total - index
                    candidate["recipe_sha256"] = digest(candidate["recipe"])
                    candidate["id"] = candidate["id"].replace(
                        "recoverable_charge_needed", "pre_shutdown_recoverable")
                    if candidate["source_snapshot_sha256"] not in energy_seen:
                        energy_rows.append(candidate)
                        energy_seen.add(candidate["source_snapshot_sha256"])
                        source_partner_counts[f"energy.{partner}"] += 1
    unfiltered_collision_count = len(collision_rows)
    unfiltered_energy_count = len(energy_rows)
    recovery_environment_steps = recovery_actions = 0
    admitted_collision, admitted_energy = [], []
    recoverability_rejections = Counter()
    for row in collision_rows:
        audit = _public_reference_recoverability(row, maximum_steps=10)
        recovery_environment_steps += audit["reference_environment_steps"]
        recovery_actions += audit["reference_actions"]
        if (audit["robot_2_active"] and audit["first_productive_step"] is not None
                and audit["longest_collision_streak"] <= 3):
            admitted_collision.append(row)
        else:
            recoverability_rejections["collision"] += 1
    for row in energy_rows:
        audit = _public_reference_recoverability(row, maximum_steps=20)
        recovery_environment_steps += audit["reference_environment_steps"]
        recovery_actions += audit["reference_actions"]
        if audit["robot_2_active"] and audit["reached_charger"]:
            admitted_energy.append(row)
        else:
            recoverability_rejections["energy"] += 1
    collision_rows, energy_rows = admitted_collision, admitted_energy
    if (not collision_rows or not energy_rows
            or digest(entries) != source_entries_sha256):
        raise ValueError("Exact v8 recovery curriculum is empty or changed train sources")
    for row in (*collision_rows, *energy_rows):
        if digest(row["snapshot"]) != row["source_snapshot_sha256"]:
            raise RuntimeError("Recovery curriculum snapshot mutated after collection")
    manifest = {
        "version": RECOVERY_CURRICULUM_VERSION,
        "source_split": "train",
        "validation_and_test_excluded": True,
        "excluded_splits": ["calibration", "validation", "extraction",
                            "explanation_test", "final_test", "play"],
        "source_actor_sha256": V8_STAGE_ACTOR_SHA256,
        "source_entries_sha256": source_entries_sha256,
        "source_scenarios": used_scenarios,
        "source_partners": list(RECOVERY_CURRICULUM_SOURCE_PARTNERS),
        "partners_called_before_actor": True,
        "source_snapshot_mutation": False,
        "action_labels_stored": False,
        "evidence_environment_steps": evidence_steps,
        "ppo_joint_steps": 0,
        "source_neural_actions": source_neural_actions,
        "source_action_overrides": 0,
        "recoverability_check": {
            "public_state_only": True,
            "action_labels_stored": False,
            "source_partner_preserved": True,
            "collision_maximum_steps": 10,
            "collision_maximum_streak": 3,
            "energy_maximum_steps": 20,
            "unfiltered_collision_count": unfiltered_collision_count,
            "unfiltered_energy_count": unfiltered_energy_count,
            "rejected": dict(recoverability_rejections),
            "environment_steps": recovery_environment_steps,
            "reference_actions": recovery_actions,
            "ppo_joint_steps": 0,
        },
        "collision_recovery_count": len(collision_rows),
        "pre_shutdown_recoverable_count": len(energy_rows),
        "shutdown_source_episodes": shutdown_episodes,
        "collision_kind_counts": dict(sorted(collision_kinds.items())),
        "source_partner_counts": dict(sorted(source_partner_counts.items())),
        "rows": [{
            "id": row["id"],
            "bucket": row["recipe"]["bucket"],
            "source_snapshot_sha256": row["source_snapshot_sha256"],
            "recipe_sha256": row["recipe_sha256"],
        } for row in (*collision_rows, *energy_rows)],
    }
    manifest["sha256"] = digest(manifest)
    return tuple(collision_rows), tuple(energy_rows), manifest


def _public_progress_options(snapshot, role=1):
    """Return legal one-step moves that reduce the current public goal distance."""
    state, config = snapshot["state"], snapshot["configuration"]
    agent = state["agents"][role]
    distance_before, charge_needed, targets = _goal_distance(snapshot, role)
    if not agent["active"] or not targets:
        return (), distance_before, charge_needed, targets
    layout = get_map_layout(config["map_layout_id"])
    options = []
    for action, delta in MOVE_DELTAS.items():
        candidate = (agent["position"][0] + delta[0],
                     agent["position"][1] + delta[1])
        if (layout.is_passable(candidate)
                and _distance_from(candidate, targets, config["map_layout_id"])
                    < distance_before):
            options.append(action)
    return tuple(options), distance_before, charge_needed, targets


def _source_action_reduces_public_goal(snapshot, action, role=1):
    """Check source action geometry without recording it in a curriculum row."""
    options, _, _, _ = _public_progress_options(snapshot, role)
    return action in options


def _hard_state_row(snapshot, *, entry, partner, category, source_frame,
                    no_progress_before):
    """Freeze an unlabeled exact v8 state selected by a public hard-state rule."""
    frozen = deepcopy(snapshot)
    frozen_sha256 = digest(frozen)
    recipe = {
        "version": HARD_STATE_CURRICULUM_VERSION,
        "source_split": "train",
        "source_scene_id": entry["id"],
        "source_scene_fingerprint": entry["fingerprint"],
        "source_partner": partner,
        "source_frame": int(source_frame),
        "category": category,
        "no_task_progress_steps_before": int(no_progress_before),
        "source_snapshot_sha256": frozen_sha256,
        "snapshot_mutation": False,
        "action_label_stored": False,
    }
    return {
        "id": (f"{entry['id']}__{partner}__frame_{int(source_frame):03d}"
               f"__{category}"),
        "snapshot": frozen,
        "source_snapshot_sha256": frozen_sha256,
        "recipe": recipe,
        "recipe_sha256": digest(recipe),
    }


def _build_v8_public_hard_state_curriculum(
        entries, actor_path, *, maximum_scenarios=128, maximum_frames=None,
        maximum_per_partner_group=HARD_STATE_MAX_PER_PARTNER_GROUP,
        environment=None):
    """Collect exact train-only v8 states for the three observed failure modes.

    Eligibility may inspect the frozen v8 command but neither that command nor
    a preferred replacement is retained.  Program partners are evaluated first
    and cannot read Actor output or mutate the source state.  Admission uses a
    separate public-state reference only to exclude unrecoverable continuations;
    its actions are also discarded.
    """
    actor_path = Path(actor_path).resolve()
    if file_hash(actor_path) != V8_STAGE_ACTOR_SHA256:
        raise ValueError("Hard-state curriculum must use the exact frozen v8 Actor")
    source_entries_sha256 = digest(entries)
    actor = NumPyNativeActor(actor_path)
    env = environment or PublicFeedbackEnvironment(
        collaborative_study_config(), PUBLIC_FEEDBACK_REWARD,
        collision_cost=.05, mode="observed",
    )
    horizon = int(maximum_frames or collaborative_study_config().horizon)
    categories = (
        "mission_wait_or_regression", "charge_needed_nonprogress",
        "long_no_task_progress",
    )
    raw = {partner: {category: [] for category in categories}
           for partner in RECOVERY_CURRICULUM_SOURCE_PARTNERS}
    seen = {partner: {category: set() for category in categories}
            for partner in RECOVERY_CURRICULUM_SOURCE_PARTNERS}
    source_counts = Counter()
    evidence_steps = source_neural_actions = 0
    used_scenarios = 0
    for scene_index, entry in enumerate(entries[:maximum_scenarios]):
        used_scenarios += 1
        for partner_index, partner in enumerate(RECOVERY_CURRICULUM_SOURCE_PARTNERS):
            reset_scenario(env, entry)
            rng = np.random.default_rng(260_911_1800 + scene_index * 101 + partner_index)
            no_progress = 0
            for _ in range(horizon):
                before = env.snapshot()
                before_sha256 = digest(before)
                state = before["state"]
                agent = state["agents"][1]
                # Avoid snapshots that cannot supply a meaningful continuation.
                remaining = int(before["configuration"]["horizon"]) - int(state["frame"])
                options, _, charge_needed, targets = _public_progress_options(before, 1)
                # This order is an enforced information-boundary invariant.
                player_action = partner_action(env, "robot_1", partner, rng)
                if player_action not in ACTIONS or digest(env.snapshot()) != before_sha256:
                    raise RuntimeError("Hard-state source partner mutated the state")
                actions, probabilities = actor.act(env.observations(), deterministic=True)
                for agent_id in env.agent_ids:
                    distribution = np.asarray(probabilities[agent_id], dtype=np.float64)
                    if actions[agent_id] != ACTIONS[int(np.argmax(distribution))]:
                        raise RuntimeError("Hard-state source action differs from v8 argmax")
                source_action = actions["robot_2"]
                reduces_goal = _source_action_reduces_public_goal(before, source_action, 1)
                eligible = []
                if agent["active"] and targets and options and remaining >= 24:
                    if charge_needed and not reduces_goal:
                        eligible.append("charge_needed_nonprogress")
                    if not charge_needed and not reduces_goal:
                        eligible.append("mission_wait_or_regression")
                    if no_progress >= HARD_STATE_NO_PROGRESS_THRESHOLD:
                        eligible.append("long_no_task_progress")
                for category in eligible:
                    if len(raw[partner][category]) >= maximum_per_partner_group:
                        continue
                    row = _hard_state_row(
                        before, entry=entry, partner=partner, category=category,
                        source_frame=state["frame"], no_progress_before=no_progress,
                    )
                    key = row["source_snapshot_sha256"]
                    if key not in seen[partner][category]:
                        raw[partner][category].append(row)
                        seen[partner][category].add(key)
                        source_counts[f"raw.{partner}.{category}"] += 1
                submitted = {"robot_1": player_action, "robot_2": source_action}
                _, _, terminated, truncated, info = env.step(submitted)
                evidence_steps += 1
                source_neural_actions += 1
                if info["requested_actions"] != submitted:
                    raise RuntimeError("Hard-state source neural action was overwritten")
                task_progress = any(event.get("event") in ("pickup", "delivery")
                                    for event in info["events"])
                no_progress = 0 if task_progress else no_progress + 1
                if terminated or truncated:
                    break

    admitted = []
    rejected = Counter()
    recovery_steps = recovery_actions = 0
    for partner in RECOVERY_CURRICULUM_SOURCE_PARTNERS:
        for category in categories:
            for row in raw[partner][category]:
                audit = _public_reference_recoverability(row, maximum_steps=20)
                recovery_steps += audit["reference_environment_steps"]
                recovery_actions += audit["reference_actions"]
                if category == "charge_needed_nonprogress":
                    valid = audit["robot_2_active"] and audit["reached_charger"]
                else:
                    valid = (audit["robot_2_active"]
                             and audit["first_productive_step"] is not None
                             and audit["first_productive_step"] <= 3)
                if valid:
                    admitted.append(row)
                    source_counts[f"admitted.{partner}.{category}"] += 1
                else:
                    rejected[f"{partner}.{category}"] += 1
    if digest(entries) != source_entries_sha256:
        raise RuntimeError("Hard-state collection mutated the train manifest")
    by_partner = Counter(row["recipe"]["source_partner"] for row in admitted)
    by_category = Counter(row["recipe"]["category"] for row in admitted)
    if (not admitted
            or any(by_partner[partner] == 0 for partner in RECOVERY_CURRICULUM_SOURCE_PARTNERS)
            or any(by_category[category] == 0 for category in categories)):
        raise ValueError("Hard-state bank lacks required partner/category coverage")
    for row in admitted:
        if digest(row["snapshot"]) != row["source_snapshot_sha256"]:
            raise RuntimeError("Hard-state snapshot changed after collection")
    manifest = {
        "version": HARD_STATE_CURRICULUM_VERSION,
        "source_split": "train",
        "validation_and_test_excluded": True,
        "excluded_splits": ["calibration", "validation", "extraction",
                            "explanation_test", "final_test", "play"],
        "source_actor_sha256": V8_STAGE_ACTOR_SHA256,
        "source_entries_sha256": source_entries_sha256,
        "source_scenarios": used_scenarios,
        "source_partners": list(RECOVERY_CURRICULUM_SOURCE_PARTNERS),
        "partners_called_before_actor": True,
        "eligibility_uses_source_actor_command": True,
        "source_actor_commands_discarded": True,
        "source_snapshot_mutation": False,
        "action_labels_stored": False,
        "categories": list(categories),
        "no_task_progress_threshold": HARD_STATE_NO_PROGRESS_THRESHOLD,
        "evidence_environment_steps": evidence_steps,
        "source_neural_actions": source_neural_actions,
        "source_action_overrides": 0,
        "ppo_joint_steps": 0,
        "raw_and_admitted_counts": dict(sorted(source_counts.items())),
        "admitted_partner_counts": dict(sorted(by_partner.items())),
        "admitted_category_counts": dict(sorted(by_category.items())),
        "recoverability_check": {
            "public_state_only": True,
            "source_partner_preserved": True,
            "action_labels_stored": False,
            "maximum_steps": 20,
            "mission_first_productive_step_max": 3,
            "rejections": dict(sorted(rejected.items())),
            "environment_steps": recovery_steps,
            "reference_actions": recovery_actions,
            "ppo_joint_steps": 0,
        },
        "rows": [{
            "id": row["id"],
            "category": row["recipe"]["category"],
            "source_partner": row["recipe"]["source_partner"],
            "source_snapshot_sha256": row["source_snapshot_sha256"],
            "recipe_sha256": row["recipe_sha256"],
        } for row in admitted],
    }
    manifest["sha256"] = digest(manifest)
    return tuple(admitted), manifest


def _actor_parameter_sha(model) -> str:
    return initialization_sha256(_cpu(model.actor.state_dict()))


def load_source(root: Path | str = SOURCE_ROOT):
    """Restore the genuine full 3.95M MPS learner through its original loader."""
    root = Path(root).resolve()
    inputs = json.loads((root / "inputs.json").read_text())
    protocols = json.loads((root / "protocols.json").read_text())
    restored, scenes, _, baseline, auxiliary, _, _, _ = source_run._material(inputs)
    trainer = source_run._trainer(
        protocols["feedback"], restored["feedback"], inputs, "feedback", auxiliary
    )
    marker = json.loads(
        (root / "branches/feedback/boundaries/step_0050000.json").read_text()
    )
    if (
        marker["actor"]["sha256"] != SOURCE_ACTOR_SHA256
        or marker["checkpoint"]["sha256"] != SOURCE_CHECKPOINT_SHA256
        or file_hash(root / marker["actor"]["path"]) != SOURCE_ACTOR_SHA256
    ):
        raise ValueError("The frozen r3 source Actor/checkpoint binding differs")
    payload = decode(root / marker["checkpoint"]["path"], SOURCE_CHECKPOINT_SHA256)
    trainer.load_state_dict(payload["trainer"])
    cumulative = trainer.source_counters["joint_steps"] + trainer.joint_steps
    if cumulative != SOURCE_CUMULATIVE_STEPS or _actor_parameter_sha(trainer.model) != marker["actor"].get(
        "parameters_sha256", _actor_parameter_sha(trainer.model)
    ):
        raise ValueError("The restored r3 learning state has a different training clock")
    return trainer, scenes, baseline


def _agent(state, role: int):
    return state["agents"][role]


def _task(state, task_id):
    return next((task for task in state["tasks"] if task["task_id"] == task_id), None)


def _goal_distance(snapshot, role: int):
    """Return public goal distance and whether charging is currently required."""
    state, config = snapshot["state"], snapshot["configuration"]
    agent = _agent(state, role)
    layout_id = config["map_layout_id"]
    distance = lambda a, b: shortest_path_distance(tuple(a), tuple(b), layout_id)
    charger = tuple(config.get("charger_position", state.get("charger_position", (5, 3))))
    # Raw snapshots retain the charger in the map layout rather than config.
    from env.warehouse.layouts import get_map_layout
    charger = get_map_layout(layout_id).charger_position
    if not agent["active"]:
        return 0, False, ()
    if agent["carrying_task_id"]:
        task = _task(state, agent["carrying_task_id"])
        targets = (tuple(task["delivery_position"]),) if task else ()
        work = min((distance(agent["position"], target) for target in targets), default=0)
        required = 2 * (work + min((distance(target, charger) for target in targets), default=0)) + 4
    else:
        tasks = [task for task in state["tasks"] if task["status"] == "available"]
        targets = tuple(tuple(task["pickup_position"]) for task in tasks)
        costs = []
        for task in tasks:
            costs.append(distance(agent["position"], task["pickup_position"])
                         + distance(task["pickup_position"], task["delivery_position"])
                         + distance(task["delivery_position"], charger))
        required = 2 * min(costs, default=0) + 4
    charge_needed = bool(targets and float(agent["battery"]) < min(100.0, required))
    effective = (tuple(charger),) if charge_needed else targets
    value = min((distance(agent["position"], target) for target in effective), default=0)
    return int(value), charge_needed, effective


def _distance_from(position, targets, layout_id):
    return min((shortest_path_distance(tuple(position), tuple(target), layout_id)
                for target in targets), default=0)


def shaping_for_transition(record, role: int):
    """Compute the registered public-state shaping for one role and one step."""
    before, after = record["before"], record["after"]
    before_state, after_state = before["state"], after["state"]
    agent_id = f"robot_{role + 1}"
    agent_before, agent_after = _agent(before_state, role), _agent(after_state, role)
    action = record["requested_actions"][agent_id]
    if not agent_before["active"]:
        return 0.0, {"productive": False, "avoidable_wait": False,
                     "distance_regression": False, "charge_needed": False}
    distance_before, charge_needed, targets = _goal_distance(before, role)
    layout_id = before["configuration"]["map_layout_id"]
    distance_after = _distance_from(agent_after["position"], targets, layout_id) if targets else distance_before
    task_progress = (
        agent_before["carrying_task_id"] != agent_after["carrying_task_id"]
        or agent_after["deliveries_completed"] > agent_before["deliveries_completed"]
    )
    # A pickup/mission transition is only an executable-task bonus when the
    # public battery budget can still finish the mission and return to the
    # charger with the configured reserve.  Moving toward the charger remains
    # productive when that budget has already been consumed.
    safe_task_progress = bool(task_progress and not charge_needed)
    productive = bool(safe_task_progress or (targets and distance_after < distance_before))
    legal_progress_exists = False
    if targets:
        from env.warehouse.layouts import get_map_layout
        layout = get_map_layout(layout_id)
        for delta in MOVE_DELTAS.values():
            candidate = (agent_before["position"][0] + delta[0],
                         agent_before["position"][1] + delta[1])
            if layout.is_passable(candidate) and _distance_from(candidate, targets, layout_id) < distance_before:
                legal_progress_exists = True
                break
    avoidable_wait = bool(action == "WAIT" and legal_progress_exists)
    regression = bool(
        not charge_needed and action != "WAIT" and targets and distance_after > distance_before
    )
    value = (SHAPING["productive_progress"] * int(productive)
             + SHAPING["avoidable_wait"] * int(avoidable_wait)
             + SHAPING["task_distance_regression"] * int(regression))
    return float(value), {
        "productive": productive,
        "safe_task_progress": safe_task_progress,
        "avoidable_wait": avoidable_wait,
        "distance_regression": regression,
        "charge_needed": charge_needed,
    }


def _critical_groups(snapshot, role):
    state, config = snapshot["state"], snapshot["configuration"]
    agents = state["agents"]
    layout_id = config["map_layout_id"]
    groups = []
    if shortest_path_distance(tuple(agents[0]["position"]), tuple(agents[1]["position"]), layout_id) <= 2:
        groups.append("narrow_passage")
    if all(agent["carrying_task_id"] is None for agent in agents) and any(
            task["status"] == "available" for task in state["tasks"]):
        groups.append("shared_pickup")
    if min(float(agent["battery"]) for agent in agents) <= 60:
        groups.append("shared_charger")
    # Every critical group must be represented by real states, but no label or
    # action is manufactured.  Sparse charger states remain tagged by geometry.
    if not groups:
        from env.warehouse.layouts import get_map_layout
        charger = get_map_layout(layout_id).charger_position
        if shortest_path_distance(tuple(agents[role]["position"]), charger, layout_id) <= 2:
            groups.append("shared_charger")
    return tuple(groups)


class ActiveTrainer(source_trainer.AlignmentTrainer):
    """Detached r4 continuation around the genuine restored r3 learner."""

    def __init__(self, source):
        source_state = source.state_dict()
        cumulative = source.source_counters["joint_steps"] + source.joint_steps
        if cumulative != SOURCE_CUMULATIVE_STEPS:
            raise ValueError("R4 must start from the exact 3.95M source")
        # The CLI owns this freshly restored object, so transferring its live
        # state avoids another expensive MPS construction without aliasing a
        # second running learner.
        self.__dict__.update(source.__dict__)
        self.source_state_sha256 = initialization_sha256(source_state)
        self.source_actor_sha256 = SOURCE_ACTOR_SHA256
        # The 3.95M alignment parent intentionally used 3e-5 for its final
        # correction stage.  R4 restores the user-specified PPO learning rate
        # for both networks while retaining the genuine Adam moments.
        self.cfg = deepcopy(self.cfg)
        self.cfg["learning_rate"] = ACTOR_LR
        self.cfg["actor_learning_rate"] = ACTOR_LR
        self.cfg["critic_learning_rate"] = CRITIC_LR
        self.cfg["entropy"] = ENTROPY_COEFFICIENT
        self.optimizers.cfg = self.cfg
        self.optimizers.actor.param_groups[0]["lr"] = ACTOR_LR
        self.optimizers.critic.param_groups[0]["lr"] = CRITIC_LR
        self.source_counters = {
            key: int(source.source_counters.get(key, 0) + getattr(source, key, 0))
            for key in ("joint_steps", "optimizer_updates", "minibatch_updates",
                        "actor_optimizer_steps", "critic_optimizer_steps")
        }
        self.source_frames = [env.state.frame for env in self.envs]
        self.training_entries = tuple(self.scenarios["splits"]["train"])
        self.training_entries_sha256 = digest(self.training_entries)
        v8_actor_path = (V8_STAGE_ROOT /
            "boundaries/step_0050000/actor.npz").resolve()
        (self.hard_state_curriculum,
         self.hard_state_curriculum_manifest) = (
            _build_v8_public_hard_state_curriculum(
                self.training_entries, v8_actor_path,
                environment=deepcopy(self.envs[0]))
        )
        self.hard_state_by_partner = {
            partner: tuple(row for row in self.hard_state_curriculum
                           if row["recipe"]["source_partner"] == partner)
            for partner in RECOVERY_CURRICULUM_SOURCE_PARTNERS
        }
        if any(not rows for rows in self.hard_state_by_partner.values()):
            raise ValueError("Every registered program partner needs hard-state rows")
        self.protocol = {
            "version": VERSION,
            "source_actor_sha256": SOURCE_ACTOR_SHA256,
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "source_state_sha256": self.source_state_sha256,
            "source_cumulative_joint_steps": SOURCE_CUMULATIVE_STEPS,
            "training": deepcopy(self.cfg),
            "partners": deepcopy(PARTNER_MIX),
            "shaping": deepcopy(SHAPING),
            "feedback": {"direction": "KL(nn||program)", "maximum_lambda": KL_MAX,
                         "applied_lambda": TRAINING_KL_LAMBDA,
                         "refresh_interval_joint_steps": 50_000,
                         "role_scope": "robot_2",
                         "training_program_minimum_fidelity": TRAINING_PROGRAM_MIN_FIDELITY,
                         "training_program_minimum_critical_fidelity": TRAINING_PROGRAM_MIN_CRITICAL_FIDELITY,
                         "final_explanation_minimum_fidelity": FINAL_EXPLANATION_MIN_FIDELITY,
                         "final_explanation_minimum_critical_fidelity": FINAL_EXPLANATION_MIN_CRITICAL_FIDELITY,
                         "runtime_action_override": False},
            "role_assignment": {
                "program_partner_role": "robot_1",
                "deployed_nn_role": "robot_2",
                "selfplay_trains_both_roles": True,
            },
            "scenario_sampling": {
                "source_split": "train",
                "ordinary_train_snapshot_probability": ORDINARY_EPISODE_PROBABILITY,
                "hard_state_snapshot_probability": HARD_STATE_EPISODE_PROBABILITY,
                "collision_recovery_snapshot_probability": COLLISION_RECOVERY_EPISODE_PROBABILITY,
                "pre_shutdown_snapshot_probability": LOW_ENERGY_EPISODE_PROBABILITY,
                "distribution": "exact_v8_public_hard_state_or_ordinary_train",
                "partner_conditioning": {
                    "partner_sampled_before_snapshot": True,
                    "curriculum_source_partner_must_equal_episode_partner": True,
                    "selfplay_always_uses_ordinary_start": True,
                    "conditional_hard_state_probability_given_program_partner": CONDITIONAL_HARD_STATE_PROBABILITY,
                    "conditional_collision_probability_given_program_partner": CONDITIONAL_COLLISION_RECOVERY_PROBABILITY,
                    "conditional_energy_probability_given_program_partner": CONDITIONAL_LOW_ENERGY_PROBABILITY,
                    "aggregate_partner_mix_unchanged": deepcopy(PARTNER_MIX),
                },
                "selection_uses_validation": False,
                "source_snapshot_mutation": False,
                "derived_snapshot_changed_fields": [],
                "action_labels_stored": False,
                "actor_observation_contract": "observed197_unchanged",
                "curriculum": deepcopy(self.hard_state_curriculum_manifest),
            },
            "maximum_additional_joint_steps": 1_000_000,
            "initial_feedback": None,
        }
        self.completed_episodes = []
        self.round_episode_prefixes = [
            _prefix(env, self.episode_returns[i], self.episode_reward_components[i])
            for i, env in enumerate(self.envs)
        ]
        for key in COUNTERS:
            setattr(self, key, 0)
        self.elapsed_seconds = 0.0
        self.feedback_enabled = False
        self.feedback_manager = FeedbackManager(self.envs[0].feature_names, TREE_CONFIG)
        self.current_lambda = 0.0
        self.feedback_rng = np.random.default_rng(260_910_401)
        self.reservoir_rng = np.random.default_rng(260_910_402)
        self.reservoir_seen = 0
        self.extraction_rows = []
        self.update_audits = []
        self.shaping_counts = Counter()
        self.action_authority = Counter()
        self.sampling_counts = Counter()
        # Do not carry a partly completed r3 episode into the new partner
        # mixture.  The optimizer/model/RNG state is genuine; each r4 rollout
        # begins at a frozen training scenario under the r4 role contract.
        for i in range(len(self.envs)):
            self.reset_one(i)

    def bind_initial_feedback(self, artifact, *, artifact_sha256, lambda_value):
        """Bind a program extracted from the exact frozen r3 source Actor.

        This is called before the first r4 PPO sample.  It cannot install a
        cross-Actor tree, and the program never supplies an environment action.
        """
        if self.joint_steps != 0 or self.current_lambda != 0:
            raise ValueError("Initial feedback can only bind at the r3 boundary")
        if (artifact.get("version") != "warehouse-r4-r3-preextraction.v1"
                or artifact.get("test_fixture") is not False
                or artifact.get("source_actor_sha256") != SOURCE_ACTOR_SHA256):
            raise ValueError("Initial RCPD artifact is not bound to the frozen r3 Actor")
        if not isinstance(artifact_sha256, str) or len(artifact_sha256) != 64:
            raise ValueError("Initial RCPD artifact SHA-256 is required")
        value = float(lambda_value)
        if not 0 < value <= KL_MAX:
            raise ValueError("Initial feedback lambda must be positive and bounded")
        self.feedback_manager.load_state_dict(artifact["feedback_manager"])
        program = self.feedback_manager.program
        if (program is None or not self.feedback_manager.reliable
                or tuple(program.feature_names) != tuple(self.envs[0].feature_names)
                or tuple(program.action_names) != tuple(ACTIONS)
                or program.metadata.get("native_source_actor_sha256") != SOURCE_ACTOR_SHA256
                or artifact.get("program_content_sha256") != digest(program.to_dict())):
            raise ValueError("Initial RCPD program/source/schema binding differs")
        self.current_lambda = value
        self.protocol["initial_feedback"] = {
            "artifact_sha256": artifact_sha256,
            "program_content_sha256": digest(program.to_dict()),
            "source_actor_sha256": SOURCE_ACTOR_SHA256,
            "lambda": value,
            "extraction_environment_steps": int(artifact["extraction_environment_steps"]),
            "ppo_joint_steps": 0,
            "runtime_action_override": False,
        }

    def bind_r4_stage_source(self, payload, *, checkpoint_sha256, actor_path):
        """Continue an admitted r4 stage with its exact Actor, Adam and RCPD."""
        actor_path = Path(actor_path).resolve()
        admitted = {
            "warehouse-r4-active-trainer.v8": {
                "actor": V8_STAGE_ACTOR_SHA256,
                "checkpoint": V8_STAGE_CHECKPOINT_SHA256,
                "steps": V8_STAGE_ADDITIONAL_STEPS,
                "lambda": TRAINING_KL_LAMBDA,
            },
            "warehouse-r4-active-trainer.v12": {
                "actor": V12_STAGE_ACTOR_SHA256,
                "checkpoint": V12_STAGE_CHECKPOINT_SHA256,
                "steps": V12_STAGE_ADDITIONAL_STEPS,
                "lambda": KL_MAX,
            },
        }
        identity = admitted.get(payload.get("version"))
        if (identity is None
                or checkpoint_sha256 != identity["checkpoint"]
                or file_hash(actor_path) != identity["actor"]
                or payload.get("joint_steps") != identity["steps"]
                or payload.get("protocol", {}).get("source_actor_sha256") != SOURCE_ACTOR_SHA256
                or payload.get("action_authority", {}).get("trainable")
                    != payload.get("action_authority", {}).get("equal")):
            raise ValueError("The r4 stage source identity or action authority differs")
        manager = FeedbackManager(self.envs[0].feature_names, TREE_CONFIG)
        manager.load_state_dict(payload["feedback_manager"])
        program = manager.program
        if (not manager.reliable or program is None
                or tuple(program.feature_names) != tuple(self.envs[0].feature_names)
                or program.metadata.get("native_source_actor_sha256") != identity["actor"]
                or float(payload.get("current_lambda", 0.0)) != KL_MAX):
            raise ValueError("The r4 stage RCPD is not bound or reliable")
        self.model.load_state_dict(payload["model"])
        self.optimizers.load_state_dict(payload["optimizers"])
        parent_actor_parameters_sha256 = NumPyNativeActor(actor_path).metadata[
            "actor_parameters_sha256"]
        parent_optimizer_sha256 = initialization_sha256(payload["optimizers"])
        parent_rng_sha256 = initialization_sha256(payload["rng"])
        parent_owned_rng_sha256 = initialization_sha256(payload["owned_rng"])
        if (_actor_parameter_sha(self.model) != parent_actor_parameters_sha256
                or initialization_sha256(_cpu(self.optimizers.state_dict()))
                    != parent_optimizer_sha256):
            raise ValueError("The v8 checkpoint and Actor parameters differ")
        # Preserve every Adam moment and step while applying this version's
        # explicitly registered learning rate for subsequent updates.
        self.optimizers.cfg = self.cfg
        self.optimizers.actor.param_groups[0]["lr"] = ACTOR_LR
        self.optimizers.critic.param_groups[0]["lr"] = CRITIC_LR
        self.rng.bit_generator.state = deepcopy(payload["rng"])
        self.feedback_rng.bit_generator.state = deepcopy(payload["feedback_rng"])
        self.reservoir_rng.bit_generator.state = deepcopy(payload["reservoir_rng"])
        self._owned_rng_state = _cpu(payload["owned_rng"])
        for key in ("joint_steps", "optimizer_updates", "minibatch_updates",
                    "actor_optimizer_steps", "critic_optimizer_steps", "episode_count"):
            setattr(self, key, int(payload[key]))
        self.feedback_manager = manager
        self.current_lambda = identity["lambda"]
        self.feedback_manager.current_lambda = identity["lambda"]
        self.protocol["initial_feedback"] = deepcopy(
            payload["protocol"].get("initial_feedback"))
        self.protocol["stage_source"] = {
            "trainer_version": payload["version"],
            "actor_sha256": identity["actor"],
            "checkpoint_sha256": identity["checkpoint"],
            "additional_ppo_joint_steps": identity["steps"],
            "program_content_sha256": digest(program.to_dict()),
            "feedback_lambda": identity["lambda"],
            "actor_and_optimizer_preserved": True,
            "parent_actor_parameters_sha256": parent_actor_parameters_sha256,
            "parent_optimizer_sha256": parent_optimizer_sha256,
            "parent_rng_sha256": parent_rng_sha256,
            "parent_owned_rng_sha256": parent_owned_rng_sha256,
            "pre_update_actor_parity": True,
            "pre_update_optimizer_parity": True,
            "optimizer_moments_and_steps_preserved": True,
            "learning_rates_rebound_after_parity": {
                "actor": ACTOR_LR, "critic": CRITIC_LR,
            },
            "rng_restored_before_curriculum_resets": True,
            "inflight_episodes_restarted_for_versioned_curriculum": True,
            "runtime_action_override": False,
        }
        self.reservoir_seen = 0
        self.extraction_rows = []
        self.update_audits = []
        self.elapsed_seconds = 0.0
        self.completed_episodes = []
        self.sampling_counts = Counter()
        self.shaping_counts = Counter(payload.get("shaping_counts", {}))
        self.action_authority = Counter(payload["action_authority"])
        for i in range(len(self.envs)):
            self.reset_one(i)

    def reset_one(self, i):
        self.partner_kinds[i] = str(self.rng.choice(
            list(PARTNER_MIX), p=list(PARTNER_MIX.values())))
        self.program_roles[i] = -1 if self.partner_kinds[i] == "selfplay" else 0
        partner = self.partner_kinds[i]
        draw = float(self.rng.random())
        if partner != "selfplay" and draw < CONDITIONAL_HARD_STATE_PROBABILITY:
            rows = self.hard_state_by_partner[partner]
            row = rows[int(self.rng.integers(len(rows)))]
            frozen_sha256 = row["source_snapshot_sha256"]
            self.envs[i].restore(row["snapshot"])
            if (digest(row["snapshot"]) != frozen_sha256
                    or digest(self.envs[i].snapshot()) != frozen_sha256
                    or not self.envs[i].state.agents[1].active):
                raise RuntimeError("Exact hard-state snapshot failed validation")
            entry_id, category = row["id"], row["recipe"]["category"]
            self.sampling_counts["hard_state_episodes"] += 1
            self.sampling_counts[f"hard_state_{category}_episodes"] += 1
            if row["recipe"]["source_partner"] != partner:
                raise RuntimeError("Hard-state curriculum partner binding differs")
        else:
            entry = self.training_entries[int(self.rng.integers(len(self.training_entries)))]
            reset_scenario(self.envs[i], entry)
            entry_id, category = entry["id"], "unchanged_train"
            self.sampling_counts["unchanged_train_episodes"] += 1
        if digest(self.training_entries) != self.training_entries_sha256:
            raise RuntimeError("The frozen train manifest changed during reset")
        initial_battery = float(self.envs[i].state.agents[1].battery)
        self.sampling_counts[f"robot_2_initial_battery_{int(initial_battery)}"] += 1
        self.scenario_ids[i] = entry_id
        # Human/program partners always occupy robot_1 during r4.  This makes
        # the deployed robot_2 the trainable neural teammate in every partner
        # episode; self-play still updates both role-conditioned policies.
        self.episode_returns[i] = 0.0
        self.episode_reward_components[i] = Counter()
        self.episode_context[i] = {
            "source": "r4_train", "curriculum_id": (
                HARD_STATE_CURRICULUM_VERSION if category != "unchanged_train" else None),
            "category": category, "start_frame": int(self.envs[i].state.frame),
            "prefix": {"team_deliveries": self.envs[i].state.total_deliveries,
                       "individual_deliveries": [a.deliveries_completed for a in self.envs[i].state.agents],
                       "shutdowns": self.envs[i].state.shutdown_count,
                       "collisions": self.envs[i].state.robot_collision_events},
        }
        self.round_episode_prefixes[i] = _prefix(self.envs[i], 0.0, {})
        self.episode_count += 1

    def _add_extraction_row(self, row):
        self.reservoir_seen += 1
        if len(self.extraction_rows) < RESERVOIR_ROWS:
            self.extraction_rows.append(row)
            return
        target = int(self.reservoir_rng.integers(self.reservoir_seen))
        if target < RESERVOIR_ROWS:
            self.extraction_rows[target] = row

    def collect(self, time_steps):
        if self.joint_steps + time_steps * len(self.envs) > 1_000_000:
            raise ValueError("R4 continuation exceeds the authorized maximum")
        batch = PublicFeedbackTrainer.collect(self, time_steps)
        # PublicFeedbackTrainer records the actual sampled NN command and the
        # exact command submitted to physics.  No later action mutation occurs.
        for record in batch["transition_records"]:
            t, i = record["rollout_index"], record["environment_index"]
            episode = f"{record['scenario_id']}:{i}:{record['before']['episode_counter']}"
            for role in range(2):
                agent_id = f"robot_{role + 1}"
                sampled = record["neural_sampled_actions"][role]
                if record["trainable"][role]:
                    self.action_authority["trainable"] += 1
                    self.action_authority["equal"] += int(record["requested_actions"][agent_id] == sampled)
                    self.action_authority[f"trainable_{agent_id}"] += 1
                    self.action_authority[f"partner_{record['partner']}_trainable_{agent_id}"] += 1
                    value, flags = shaping_for_transition(record, role)
                    base_value = float(batch["rewards"][t, i, role])
                    batch["rewards"][t, i, role] += value
                    self.shaping_counts["base_reward_sum"] += base_value
                    self.shaping_counts["base_reward_abs_sum"] += abs(base_value)
                    self.shaping_counts["shaping_reward_sum"] += value
                    self.shaping_counts["shaping_reward_abs_sum"] += abs(value)
                    for name, active in flags.items():
                        self.shaping_counts[name] += int(active)
                    self.shaping_counts["rows"] += 1
                    obs = np.asarray(batch["observations"][t, i, role], dtype=np.float32).copy()
                    self._add_extraction_row({"observation": obs, "episode": episode,
                        "groups": _critical_groups(record["before"], role), "role": role})
                if record["requested_actions"][agent_id] != sampled and record["trainable"][role]:
                    raise RuntimeError("A trainable r4 neural action was overwritten")
        _, state = self.arrays()
        with self._rng_context():
            _, bootstrap = self.infer(*self.arrays())
        batch["advantages"], batch["returns"] = gae(
            batch["rewards"], batch["values"], batch["dones"], bootstrap,
            self.cfg["gamma"], self.cfg["gae_lambda"],
        )
        return batch

    def native_ppo_update(self, batch):
        result = NativeV2Trainer.update(self, batch)
        feedback_update._sync_steps(self)
        return result

    def update(self, batch):
        result = feedback_update.stable_ppo_update(
            self, batch, self.feedback_manager.program,
            lambda_value=self.current_lambda, auxiliary_observations=None,
            feedback_role=1,
        )
        feedback_update._sync_steps(self)
        audit = result.get("stable_update", {})
        self.update_audits.append({"joint_steps": self.joint_steps,
            "lambda": self.current_lambda, "path": audit.get("path"),
            "actor_adam_steps": audit.get("actor_adam_steps"),
            "critic_adam_steps": audit.get("critic_adam_steps"),
            "feedback_loss": result.get("feedback_loss", 0.0)})
        return result

    def fit_program(self, actor_file_sha256):
        if len(self.extraction_rows) < 512:
            raise ValueError("Insufficient real neural trajectory rows for RCPD")
        # Keep one exact observation in only one split.  The split unit remains
        # the actual episode, and all probabilities are re-evaluated with the
        # frozen boundary Actor rather than stale rollout logits.
        unique, fingerprints = [], set()
        for row in self.extraction_rows:
            if row.get("role") != 1:
                continue
            fingerprint = sha256(row["observation"].astype("<f4", copy=False).tobytes()).hexdigest()
            if fingerprint in fingerprints:
                continue
            fingerprints.add(fingerprint); unique.append(row)
        train, validation = [], []
        for row in unique:
            bucket = int(sha256(row["episode"].encode()).hexdigest()[:8], 16) % 5
            (validation if bucket == 0 else train).append(row)
        if len(train) < 256 or len(validation) < 128:
            raise ValueError("Episode-disjoint RCPD split is too small")
        def arrays(rows):
            obs = np.stack([row["observation"] for row in rows]).astype(np.float32)
            with torch.no_grad():
                logits = self.model.actor_logits(torch.as_tensor(obs, device=self.device))
                probs = torch.softmax(logits, -1).cpu().numpy().astype(np.float32)
            return obs, probs, [row["episode"] for row in rows], [row["groups"] for row in rows]
        tx, ty, tids, tgroups = arrays(train)
        vx, vy, vids, vgroups = arrays(validation)
        manager = FeedbackManager(self.envs[0].feature_names, TREE_CONFIG)
        report = manager.fit(tx, ty, vx, vy,
            step=SOURCE_CUMULATIVE_STEPS + self.joint_steps,
            source_actor_sha256=actor_file_sha256, train_episode_ids=tids,
            val_episode_ids=vids, train_groups=tgroups, val_groups=vgroups)
        # Generic RCPD enumerates names alphabetically. Rebind to the Actor's
        # exact observed197 order; predicates remain name-based.
        if manager.program is not None:
            manager.program = ExecutableProgram(
                manager.program.action_names,
                tuple(self.envs[0].feature_names),
                manager.program.root,
                {**dict(manager.program.metadata), "feature_order_rebound_to_actor": True,
                 "role_scope": ["robot_2"]},
            )
        selected = report["selected"]
        report["final_explanation_gate"] = {
            "minimum_fidelity": FINAL_EXPLANATION_MIN_FIDELITY,
            "minimum_critical_fidelity": FINAL_EXPLANATION_MIN_CRITICAL_FIDELITY,
            "passed": bool(
                selected["fidelity"] >= FINAL_EXPLANATION_MIN_FIDELITY
                and all(item["fidelity"] is not None
                        and item["fidelity"] >= FINAL_EXPLANATION_MIN_CRITICAL_FIDELITY
                        for item in selected["critical"].values())
            ),
            "separate_from_training_regularizer_admission": True,
        }
        self.feedback_manager = manager
        # An admitted current-Actor program applies a nonzero bounded KL during
        # the next cycle.  Failed fits are retained but exert zero gradient.
        self.current_lambda = TRAINING_KL_LAMBDA if manager.reliable else 0.0
        self.extraction_rows = []
        self.reservoir_seen = 0
        return report

    def state_dict_r4(self):
        return {
            "version": VERSION, "protocol": deepcopy(self.protocol),
            "source_state_sha256": self.source_state_sha256,
            "source_counters": deepcopy(self.source_counters),
            "model": _cpu(self.model.state_dict()),
            "optimizers": _cpu(self.optimizers.state_dict()),
            "rng": deepcopy(self.rng.bit_generator.state),
            "feedback_rng": deepcopy(self.feedback_rng.bit_generator.state),
            "reservoir_rng": deepcopy(self.reservoir_rng.bit_generator.state),
            "owned_rng": _cpu(self._owned_rng_state),
            "envs": [env.snapshot() for env in self.envs],
            "feedback_manager": self.feedback_manager.state_dict(),
            "current_lambda": self.current_lambda,
            "reservoir_seen": self.reservoir_seen,
            "extraction_rows": deepcopy(self.extraction_rows),
            "update_audits": deepcopy(self.update_audits),
            "shaping_counts": dict(self.shaping_counts),
            "action_authority": dict(self.action_authority),
            "sampling_counts": dict(self.sampling_counts),
            "elapsed_seconds": self.elapsed_seconds,
            "round_episode_prefixes": deepcopy(self.round_episode_prefixes),
            "completed_episodes": _cpu(self.completed_episodes),
            "episode_reward_components": [dict(x) for x in self.episode_reward_components],
            **{key: _cpu(getattr(self, key)) for key in (*COUNTERS, *EPISODE_FIELDS)},
        }

    def load_state_dict_r4(self, payload):
        if payload.get("version") != VERSION or payload.get("protocol") != self.protocol:
            raise ValueError("R4 checkpoint protocol differs")
        self.model.load_state_dict(payload["model"])
        self.optimizers.load_state_dict(payload["optimizers"])
        self.rng.bit_generator.state = deepcopy(payload["rng"])
        self.feedback_rng.bit_generator.state = deepcopy(payload["feedback_rng"])
        self.reservoir_rng.bit_generator.state = deepcopy(payload["reservoir_rng"])
        self._owned_rng_state = _cpu(payload["owned_rng"])
        for env, snapshot in zip(self.envs, payload["envs"]):
            env.restore(snapshot, require_feedback=True, require_credit=True, require_shutdown=True)
        self.feedback_manager.load_state_dict(payload["feedback_manager"])
        for key in ("current_lambda", "reservoir_seen", "extraction_rows", "update_audits",
                    "elapsed_seconds", "round_episode_prefixes", "completed_episodes"):
            setattr(self, key, deepcopy(payload[key]))
        self.shaping_counts = Counter(payload["shaping_counts"])
        self.action_authority = Counter(payload["action_authority"])
        self.sampling_counts = Counter(payload["sampling_counts"])
        self.episode_reward_components = [Counter(x) for x in payload["episode_reward_components"]]
        for key in (*COUNTERS, *EPISODE_FIELDS):
            setattr(self, key, deepcopy(payload[key]))

    def export(self, path):
        metadata = {
            "experiment_version": VERSION,
            "joint_steps": self.joint_steps,
            "cumulative_joint_steps": SOURCE_CUMULATIVE_STEPS + self.joint_steps,
            "optimizer_updates": self.optimizer_updates,
            "actor_parameters_sha256": _actor_parameter_sha(self.model),
            "source_actor_sha256": SOURCE_ACTOR_SHA256,
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "source_counters": deepcopy(self.source_counters),
            "protocol_sha256": digest(self.protocol),
            "public_feedback_version": "warehouse-native-public-feedback.v1",
            "public_feedback_mode": "observed",
            "feature_names": list(self.envs[0].feature_names),
            "feedback_lambda": self.current_lambda,
            "action_controller": "neural_actor_only",
            "runtime_action_override": False,
            "candidate": True,
        }
        return self.model.export_npz(path, metadata)
