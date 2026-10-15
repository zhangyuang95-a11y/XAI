"""Deterministic, contribution-aware evaluation for the public-history Actor.

This module loads no models or scenario files. Callers provide a bound Actor and
the existing validation entries, and own any persistent evaluation reservation.
Program actions are chosen before neural inference. Only frozen physics can
cancel a submitted command; there is no action mask, repair or runtime Critic.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
import re

import numpy as np

from backend.training.warehouse_native_common import digest, jsonable
from backend.training.warehouse_native_evaluation import capability, summarize
from backend.training.warehouse_native_feedback_run import warmup_capability
from backend.training.warehouse_native_public_feedback import (
    PublicFeedbackEnvironment, VERSION as OBSERVER_VERSION,
    OBSERVATION_SIZE, GLOBAL_STATE_SIZE, MODES, HISTORY_FEATURE_NAMES,
)
from env.warehouse.domain import collaborative_study_config
from env.warehouse.navigation import ACTIONS, MOVE_DELTAS
from env.warehouse_native.observations import observation_names
from env.warehouse_native.partners import partner_action
from env.warehouse_native.policy import NATIVE_ACTOR_FORMAT, NATIVE_POLICY_VERSION
from env.warehouse_native.scenarios import reset_scenario

VERSION = "warehouse-native-public-feedback-evaluation.v1"
PARTNERS = ("skilled", "assertive", "noisy")
PHYSICAL_TRAJECTORY_KEYS = ("before", "after", "requested_actions", "executed_actions", "events")
REQUIRED_BINDINGS = ("protocol_sha256", "scenario_manifest_sha256", "initialization_sha256", "source_sha256", "joint_steps")
ABSOLUTE_GATE = {
    "episodes_per_partner": 50, "minimum_skilled_delivery_ratio_to_reference": .8,
    "minimum_noisy_delivery_ratio_to_reference": .6, "minimum_ai_deliveries_mean": 2.,
    "minimum_ai_active_end_rate": .9, "minimum_delivery_improvement_over_random": .2,
}
WARMUP_GATE = {"minimum_joint_steps": 100000, "minimum_ai_deliveries_mean": 2.,
    "minimum_ai_active_end_rate": .9, "minimum_delivery_ratio_to_reference": .7}
REWARD = {"version": "warehouse-native-score-pbrs.v2", "native_score_scale": .01,
    "static_wall_command_cost": .02, "potential_scale": .25, "gamma": .99,
    "shared_reward": True, "pickup_bonus": 0., "charge_bonus": 0.}
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _sha(value):
    return isinstance(value, str) and bool(_SHA.fullmatch(value))


def _contract(actor, mode, bindings, protocol, fixture, config):
    if type(fixture) is not bool or mode not in MODES or type(mode) is not str:
        raise ValueError("Evaluation mode and fixture flag differ from their contract")
    if not isinstance(protocol, dict) or not isinstance(bindings, dict) or not set(REQUIRED_BINDINGS) <= set(bindings):
        raise ValueError("Evaluation requires externally retained protocol and Actor bindings")
    if protocol.get("test_fixture", False) is not fixture:
        raise ValueError("Protocol fixture provenance differs")
    if protocol.get("absolute_gate") != ABSOLUTE_GATE or protocol.get("warmup_gate") != WARMUP_GATE:
        raise ValueError("The frozen absolute and warmup gates cannot be changed")
    evaluation = protocol.get("evaluation", {})
    if (evaluation.get("partners") != list(PARTNERS) or evaluation.get("deterministic") is not True
            or evaluation.get("episodes_per_partner") != 50 or evaluation.get("read_final_test") is not False):
        raise ValueError("Evaluation must retain the registered development partners and deterministic scope")
    if (protocol.get("public_feedback_version") != OBSERVER_VERSION
            or protocol.get("collision_training_cost") != .05 or protocol.get("reward") != REWARD):
        raise ValueError("Observer or reward revision differs")
    step = bindings["joint_steps"]
    if type(step) is not int or step < 0 or (not fixture and step not in (0, 50000, 100000)):
        raise ValueError("Evaluation step must be an existing registered continuation boundary")
    for name in REQUIRED_BINDINGS[:-1]:
        if not _sha(bindings[name]): raise ValueError("Invalid expected Actor hash binding: " + name)
    metadata = getattr(actor, "metadata", None)
    if not isinstance(metadata, dict) or not callable(getattr(actor, "act", None)):
        raise ValueError("A metadata-bound native Actor is required; no program fallback is accepted")
    required = {"format": NATIVE_ACTOR_FORMAT, "policy_version": NATIVE_POLICY_VERSION,
        "obs_dim": OBSERVATION_SIZE, "state_dim": GLOBAL_STATE_SIZE, "hidden": 128,
        "architecture": "two_hidden_layer_tanh", "actions": list(ACTIONS),
        "action_masks": False, "runtime_action_override": False,
        "public_feedback_version": OBSERVER_VERSION, "public_feedback_mode": mode,
        "feature_names": list(observation_names(config)) + list(HISTORY_FEATURE_NAMES)}
    for name, value in required.items():
        if metadata.get(name) != value or (name in ("action_masks", "runtime_action_override") and metadata.get(name) is not False):
            raise ValueError("Actor metadata differs: " + name)
    if any(type(metadata.get(name)) is not int for name in ("obs_dim", "state_dim", "hidden")):
        raise ValueError("Actor dimensions must be integer contracts")
    if metadata.get("test_fixture", False) is not fixture:
        raise ValueError("Actor fixture provenance differs")
    for name, value in bindings.items():
        got = getattr(actor, "artifact_sha256", None) if name == "actor_sha256" else metadata.get(name)
        if name == "actor_sha256" and not _sha(value): raise ValueError("Invalid Actor artifact hash")
        if got != value: raise ValueError("Actor binding differs: " + name)
    source_steps = protocol.get("initialization", {}).get("source_joint_steps")
    retained_source = metadata.get("source_counters", {}).get("joint_steps")
    if (type(source_steps) is not int or source_steps < 0
            or (not fixture and source_steps != 280000)
            or type(retained_source) is not int or retained_source != source_steps):
        raise ValueError("Actor inherited PPO counter differs from the prescribed initialization")
    if type(metadata.get("joint_steps")) is not int:
        raise ValueError("Actor continuation step must be an integer")
    if not fixture and config != collaborative_study_config():
        raise ValueError("Production evaluation cannot alter the frozen environment configuration")
    if type(config.horizon) is not int or not 1 <= config.horizon <= 120:
        raise ValueError("Evaluation horizon must be a positive integer no larger than120")
    return deepcopy(metadata)


def _scenarios(scenarios, fixture, config, *, start_index=0, single=False):
    # A whole manifest is deliberately rejected: this evaluator never selects or
    # opens final_test, including in test-fixture mode.
    if not isinstance(scenarios, (list, tuple)) or not scenarios or (not fixture and len(scenarios) != (1 if single else 50)):
        raise ValueError("Pass only the existing ordered validation entries")
    ids, fingerprints = [], []
    for index, scene in enumerate(scenarios, start=start_index):
        if not isinstance(scene, dict) or scene.get("id") != f"validation_{index:04d}" or scene.get("split", "validation") != "validation":
            raise ValueError("Only the original ordered validation split is permitted")
        if scene.get("test_fixture", False) is not fixture or not _sha(scene.get("fingerprint")):
            raise ValueError("Scenario fixture provenance/fingerprint differs")
        snapshot = scene.get("snapshot", {})
        if (not isinstance(snapshot, dict) or any(str(key).startswith("public_feedback") for key in snapshot)
                or snapshot.get("configuration") != asdict(config)):
            raise ValueError("Validation starts must be raw frozen-configuration snapshots")
        state = snapshot.get("state", {})
        if (type(state.get("frame")) is not int or state["frame"] != 0
                or state.get("terminated") is not False or state.get("truncated") is not False
                or state.get("total_deliveries") != 0 or state.get("robot_collision_events") != 0
                or state.get("shutdown_count") != 0 or state.get("user_score") != 0):
            raise ValueError("Each validation execution starts from its original unfinished frame0")
        agents = state.get("agents", [])
        if len(agents) != 2 or any(a.get("active") is not True or a.get("deliveries_completed") != 0 for a in agents):
            raise ValueError("Initial role state differs from the validation contract")
        ids.append(scene["id"]); fingerprints.append(scene["fingerprint"])
    if len(set(ids)) != len(ids) or len(set(fingerprints)) != len(fingerprints):
        raise ValueError("Validation IDs and physical fingerprints must be unique")
    return deepcopy(list(scenarios))


def _baseline(report, scenes):
    if not isinstance(report, dict) or not isinstance(report.get("rows"), list):
        raise ValueError("Baseline requires retained per-scenario records")
    expected = {scene["id"]: scene["fingerprint"] for scene in scenes}
    if len(report["rows"]) != len(scenes) * len(PARTNERS):
        raise ValueError("Baseline sample count differs")
    for partner in PARTNERS:
        rows = [r for r in report["rows"] if r.get("partner") == partner]
        if (len(rows) != len(scenes) or {r.get("scenario_id"): r.get("initial_fingerprint") for r in rows} != expected
                or any(r.get("neural_role") != 1 for r in rows)):
            raise ValueError("Baseline does not cover the same scenarios and neural role")
        calculated = summarize(rows)
        if any(report.get("summary", {}).get(partner, {}).get(k) != v for k, v in calculated.items()):
            raise ValueError("Baseline aggregates differ from its retained rows")
    return deepcopy(report)


def _outputs(actor, observations):
    proposals, probabilities = actor.act(observations, deterministic=True)
    if not isinstance(proposals, dict) or not isinstance(probabilities, dict) or set(proposals) != set(observations) or set(probabilities) != set(observations):
        raise ValueError("Actor must return both raw commands and complete distributions")
    for key in observations:
        p = np.asarray(probabilities[key])
        if (p.shape != (5,) or not np.isfinite(p).all() or (p < 0).any()
                or abs(float(p.sum()) - 1.) > 1e-6 or proposals[key] != ACTIONS[int(np.argmax(p))]):
            raise ValueError("Actor command is not its unchanged deterministic distribution argmax")
    return deepcopy(proposals), {key: np.asarray(value).copy() for key, value in probabilities.items()}


def _callback(callback, value):
    if callback is not None: callback(deepcopy(value))


def _episode(actor, scene, partner, mode, protocol, config, context, before_step, on_step, keep_trace):
    env = PublicFeedbackEnvironment(config, protocol["reward"], collision_cost=.05, mode=mode)
    reset_scenario(env, scene)
    if env.public_history()["valid"] or any(env.public_history()["consecutive_move_canceled"].values()):
        raise ValueError("Validation must begin without invented action history")
    rng = np.random.default_rng(np.random.SeedSequence([context["seed"], 202]))
    waits, blocked, gain, charge, walls = [0, 0], [0, 0], [0., 0.], [0, 0], [0, 0]
    counts, kinds, configuration_counts = Counter(), Counter(), Counter()
    streak = no_progress = longest = longest_no_progress = 0
    trace, physical_trace = [], []
    while not env.done:
        before = env.public_view(); snapshot_hash = digest(env.snapshot())
        obs = env.observations()
        # The program has no learner-output argument and decides first.
        program = partner_action(env, "robot_1", partner, rng)
        if program not in ACTIONS or digest(env.snapshot()) != snapshot_hash:
            raise ValueError("Program partner changed the environment or returned an invalid command")
        proposals, probabilities = _outputs(actor, obs)
        if digest(env.snapshot()) != snapshot_hash:
            raise ValueError("Actor inference changed the current environment")
        submitted = {"robot_1": program, "robot_2": proposals["robot_2"]}
        step_context = {**context, "frame": before["frame"], "actual_steps_completed": counts["steps"]}
        _callback(before_step, step_context)
        _, rewards, terminated, truncated, info = env.step(submitted)
        counts["steps"] += 1
        if info["requested_actions"] != submitted or submitted["robot_2"] != proposals["robot_2"]:
            raise RuntimeError("Neural command was overwritten before physical resolution")
        after = env.public_view()
        collision = bool(info["robot_collision"])
        streak = streak + 1 if collision else 0; longest = max(longest, streak)
        task_progress = any(event["event"] in ("pickup", "delivery") for event in info["events"])
        no_progress = 0 if task_progress else no_progress + 1
        longest_no_progress = max(longest_no_progress, no_progress)
        counts["collisions"] += collision
        if collision: kinds[info["collision_kind"]] += 1
        for i, key in enumerate(env.agent_ids):
            waits[i] += submitted[key] == "WAIT"
            blocked[i] += submitted[key] != "WAIT" and info["executed_actions"][key] == "WAIT"
            walls[i] += key in info["invalid_moves"]
            amount = max(0., after["agents"][i]["battery"] - before["agents"][i]["battery"])
            gain[i] += amount; charge[i] += amount > 0
        if before["agents"][1]["active"]:
            counts["active_neural_commands"] += 1
            command = proposals["robot_2"]
            counts["neural_static_wall_commands"] += command in MOVE_DELTAS and obs["robot_2"][env.feature_names.index(f"self.neighbor.{command}.passable")] < .5
        configuration_counts[digest({"positions": [a["position"] for a in after["agents"]],
            "cargo": [a["carrying_task_id"] for a in after["agents"]],
            "tasks": [(t["task_id"], t["status"]) for t in after["tasks"]], "delivered": after["total_deliveries"]})] += 1
        record = {**step_context, "before": before, "after": after,
            "snapshot_before_sha256": snapshot_hash, "snapshot_after_sha256": digest(env.snapshot()),
            "policy_actions": proposals, "program_action": program, "submitted_actions": submitted,
            "executed_actions": info["executed_actions"], "intended_targets": info["intended_targets"],
            "action_distributions": {key: value.tolist() for key, value in probabilities.items()},
            "actor_observations": {key: value.tolist() for key, value in obs.items()},
            "events": info["events"], "invalid_moves": info["invalid_moves"],
            "robot_collision": collision, "collision_kind": info["collision_kind"],
            "rewards": rewards, "reward_components": info["reward_components"],
            "nn_action_overrides": 0, "terminated": terminated, "truncated": truncated}
        record = jsonable(record)
        physical_trace.append({"before": record["before"], "after": record["after"],
            "requested_actions": record["submitted_actions"], "executed_actions": record["executed_actions"],
            "events": record["events"]})
        _callback(on_step, record)
        if keep_trace: trace.append(record)
        if counts["steps"] > config.horizon: raise RuntimeError("Evaluation exceeded the episode limit")
    s = env.state; steps = counts["steps"]
    row = {**context, "initial_fingerprint": scene["fingerprint"], "neural_role": 1,
        "source_joint_steps": protocol["initialization"]["source_joint_steps"],
        "additional_joint_steps": context["actor_bindings"]["joint_steps"],
        "total_actor_training_joint_steps": protocol["initialization"]["source_joint_steps"] + context["actor_bindings"]["joint_steps"],
        "steps": steps, "actual_environment_steps": steps, "team_deliveries": s.total_deliveries,
        "individual_deliveries": [a.deliveries_completed for a in s.agents],
        "ai_deliveries": s.agents[1].deliveries_completed, "program_deliveries": s.agents[0].deliveries_completed,
        "ai_active_end": bool(s.agents[1].active), "collisions": s.robot_collision_events,
        "shutdowns": s.shutdown_count, "native_score": float(s.user_score), "legacy_score": None,
        "legacy_score_status": "not_replayed_against_original_planner", "waits": waits,
        "blocked_moves": blocked, "charge_gain": gain, "charging_steps": charge,
        "wall_invalid_by_role": walls, "invalid_moves": s.invalid_move_count,
        "max_repeated_configuration": max(configuration_counts.values()), "terminal_reason": s.terminal_reason,
        "nn_action_overrides": 0, "raw_neural_actions_submitted": steps, "active_neural_commands": counts["active_neural_commands"],
        "neural_static_wall_commands": counts["neural_static_wall_commands"],
        "neural_static_wall_rate": counts["neural_static_wall_commands"] / max(1, counts["active_neural_commands"]),
        "collisions_per_step": s.robot_collision_events / steps, "collision_kinds": dict(kinds),
        "longest_consecutive_collisions": longest, "longest_steps_without_task_progress": longest_no_progress,
        "no_progress_definition": "Consecutive steps without pickup or delivery; diagnostic only, not an error or reward.",
        "final_public_history": env.public_history(),
        "physical_trajectory_sha256": digest(physical_trace),
        "physical_trajectory_keys": list(PHYSICAL_TRAJECTORY_KEYS)}
    if keep_trace:
        row["trace"] = trace
        row["physical_trace"] = physical_trace
    return jsonable(row)


def _options(seed, keep_trace, callbacks):
    if type(seed) is not int or seed != 17000 or type(keep_trace) is not bool:
        raise ValueError("Keep the original evaluation seed17000 and explicit trace flag")
    if any(callback is not None and not callable(callback) for callback in callbacks):
        raise ValueError("Evaluation callback must be callable")


def _context(scene, partner, mode, index, count, seed, config, fixture, bindings):
    return {"evaluation_version": VERSION, "mode": mode, "partner": partner,
        "scenario_id": scene["id"], "episode_index": PARTNERS.index(partner) * count + index,
        "seed": seed + index, "horizon": config.horizon,
        "maximum_environment_steps": config.horizon,
        "test_fixture": fixture, "actor_bindings": deepcopy(bindings)}


def evaluate_episode(actor, scenario, partner, mode, *, expected_bindings, protocol,
                     scenario_index, scenario_count=50, seed=17000,
                     before_episode=None, before_step=None, on_step=None, on_episode=None,
                     keep_trace=False, allow_test_fixture=False, config=None):
    """Execute exactly one original episode; the caller persists its receipt.

    The caller must authenticate retained rows against its committed audit ledger
    before passing them to ``assemble_report``. No checkpoint or row files are
    opened here. A failed hook propagates; this module never retries a trajectory.
    """
    _options(seed, keep_trace, (before_episode, before_step, on_step, on_episode))
    config = config or collaborative_study_config()
    _contract(actor, mode, expected_bindings, protocol, allow_test_fixture, config)
    if (type(scenario_count) is not int or scenario_count < 1
            or (not allow_test_fixture and scenario_count != 50)
            or type(scenario_index) is not int or not 0 <= scenario_index < scenario_count
            or partner not in PARTNERS):
        raise ValueError("Episode is outside the ordered original evaluation matrix")
    scene = _scenarios([scenario], allow_test_fixture, config, start_index=scenario_index, single=True)[0]
    context = _context(scene, partner, mode, scenario_index, scenario_count, seed,
                       config, allow_test_fixture, expected_bindings)
    _callback(before_episode, context)
    row = _episode(actor, scene, partner, mode, protocol, config, context, before_step, on_step, keep_trace)
    _callback(on_episode, row)
    return row


def evaluate_actor(actor, scenarios, mode, *, expected_bindings, protocol,
                   reference_report=None, random_report=None, seed=17000,
                   before_episode=None, before_step=None, on_step=None, on_episode=None,
                   keep_trace=False, allow_test_fixture=False, config=None):
    """Run the full matrix. Use evaluate_episode for crash-resumable execution.

    before_episode runs before restore, before_step before the actual transition.
    Completed hooks receive detached data. Reservations are never refunded here.
    """
    _options(seed, keep_trace, (before_episode, before_step, on_step, on_episode))
    config = config or collaborative_study_config()
    _contract(actor, mode, expected_bindings, protocol, allow_test_fixture, config)
    scenes = _scenarios(scenarios, allow_test_fixture, config)
    if (reference_report is None) != (random_report is None):
        raise ValueError("Supply both retained baselines or neither")
    if reference_report is not None:
        _baseline(reference_report, scenes); _baseline(random_report, scenes)
    rows = []
    for partner in PARTNERS:
        for index, scene in enumerate(scenes):
            rows.append(evaluate_episode(actor, scene, partner, mode,
                expected_bindings=expected_bindings, protocol=protocol,
                scenario_index=index, scenario_count=len(scenes), seed=seed,
                before_episode=before_episode, before_step=before_step, on_step=on_step,
                on_episode=on_episode, keep_trace=keep_trace,
                allow_test_fixture=allow_test_fixture, config=config))
    return assemble_report(rows, scenes, mode, expected_bindings=expected_bindings,
        protocol=protocol, actor_metadata=actor.metadata, reference_report=reference_report,
        random_report=random_report, allow_test_fixture=allow_test_fixture, config=config)


def assemble_report(rows, scenarios, mode, *, expected_bindings, protocol, actor_metadata,
                    reference_report=None, random_report=None, allow_test_fixture=False, config=None):
    """Pure aggregation of caller-authenticated, committed episode receipts.

    Identity and numerical contracts are checked here, not receipt authenticity.
    The runner must bind each retained row to its separately persisted reservation
    and commit proof. This function performs zero Actor or environment operations.
    """
    config = config or collaborative_study_config()
    contract_actor = SimpleNamespace(metadata=actor_metadata, act=lambda *a, **k: None,
        artifact_sha256=expected_bindings.get("actor_sha256"))
    metadata = _contract(contract_actor, mode, expected_bindings, protocol, allow_test_fixture, config)
    scenes = _scenarios(scenarios, allow_test_fixture, config)
    if (reference_report is None) != (random_report is None):
        raise ValueError("Supply both retained baselines or neither")
    reference = _baseline(reference_report, scenes) if reference_report is not None else None
    random = _baseline(random_report, scenes) if random_report is not None else None
    if not isinstance(rows, list) or len(rows) != len(scenes) * len(PARTNERS):
        raise ValueError("Aggregation requires the complete committed episode matrix")
    for pindex, partner in enumerate(PARTNERS):
        for index, scene in enumerate(scenes):
            row = rows[pindex * len(scenes) + index]
            context = _context(scene, partner, mode, index, len(scenes), 17000,
                               config, allow_test_fixture, expected_bindings)
            if (not isinstance(row, dict) or any(row.get(k) != v for k, v in context.items())
                    or row.get("initial_fingerprint") != scene["fingerprint"]
                    or row.get("neural_role") != 1 or row.get("nn_action_overrides") != 0
                    or not _sha(row.get("physical_trajectory_sha256"))
                    or row.get("physical_trajectory_keys") != list(PHYSICAL_TRAJECTORY_KEYS)):
                raise ValueError("Committed episode identity or raw-action contract differs")
            inherited = metadata["source_counters"]["joint_steps"]
            if (row.get("source_joint_steps") != inherited
                    or row.get("additional_joint_steps") != expected_bindings["joint_steps"]
                    or row.get("total_actor_training_joint_steps") != inherited + expected_bindings["joint_steps"]):
                raise ValueError("Committed episode source/additional PPO counters differ")
            steps = row.get("steps")
            if (type(steps) is not int or not 1 <= steps <= config.horizon
                    or row.get("actual_environment_steps") != steps):
                raise ValueError("Committed episode step accounting differs")
            if "physical_trace" in row and (len(row["physical_trace"]) != steps
                    or any(set(frame) != set(PHYSICAL_TRAJECTORY_KEYS) for frame in row["physical_trace"])
                    or digest(row["physical_trace"]) != row["physical_trajectory_sha256"]):
                raise ValueError("Committed physical trace differs from its comparison hash")
            for field in ("team_deliveries", "ai_deliveries", "program_deliveries", "collisions",
                          "shutdowns", "invalid_moves", "active_neural_commands", "neural_static_wall_commands",
                          "longest_consecutive_collisions", "longest_steps_without_task_progress"):
                if type(row.get(field)) is not int or row[field] < 0:
                    raise ValueError("Invalid committed episode count: " + field)
            if (row["individual_deliveries"] != [row["program_deliveries"], row["ai_deliveries"]]
                    or row["team_deliveries"] != sum(row["individual_deliveries"])
                    or not 0 <= row["neural_static_wall_commands"] <= row["active_neural_commands"] <= steps
                    or not 0 <= row["longest_consecutive_collisions"] <= row["collisions"] <= steps
                    or row["longest_steps_without_task_progress"] > steps
                    or type(row.get("ai_active_end")) is not bool
                    or row["collisions_per_step"] != row["collisions"] / steps):
                raise ValueError("Committed episode metrics are internally inconsistent")
    rows = deepcopy(rows)
    summaries = {}
    for partner in PARTNERS:
        selected = [r for r in rows if r["partner"] == partner]
        summary = summarize(selected); commands = sum(r["active_neural_commands"] for r in selected)
        wall = sum(r["neural_static_wall_commands"] for r in selected); steps = sum(r["steps"] for r in selected)
        summary.update(active_neural_commands=commands, neural_static_wall_commands=wall,
            neural_static_wall_rate=wall / max(1, commands), environment_steps=steps,
            collisions_per_step=sum(r["collisions"] for r in selected) / steps,
            mean_longest_consecutive_collisions=float(np.mean([r["longest_consecutive_collisions"] for r in selected])),
            max_consecutive_collisions=max(r["longest_consecutive_collisions"] for r in selected),
            mean_longest_steps_without_task_progress=float(np.mean([r["longest_steps_without_task_progress"] for r in selected])),
            mean_program_deliveries=float(np.mean([r["program_deliveries"] for r in selected])))
        summaries[partner] = summary
    report = {"version": VERSION, "summary": summaries, "rows": rows,
        "metric": "mean_team_deliveries", "primary_metric": "equal_partner_mean_of_nn_deliveries",
        "primary_value": float(np.mean([s["mean_ai_deliveries"] for s in summaries.values()])),
        "deterministic_actor": True, "mode": mode, "public_feedback_version": OBSERVER_VERSION,
        "actor_bindings": deepcopy(expected_bindings), "actor_metadata_sha256": digest(metadata),
        "source_joint_steps": metadata["source_counters"]["joint_steps"],
        "additional_joint_steps": expected_bindings["joint_steps"],
        "total_actor_training_joint_steps": metadata["source_counters"]["joint_steps"] + expected_bindings["joint_steps"],
        "validation_entries_sha256": digest(scenes), "episodes": len(rows),
        "environment_steps": sum(r["steps"] for r in rows), "training_steps": 0, "optimizer_updates": 0,
        "nn_action_overrides": 0, "raw_neural_actions_submitted": sum(r["steps"] for r in rows),
        "test_fixture": allow_test_fixture,
        "scope": "existing_development_validation", "final_test_read": False,
        "formal_ready": False, "qualification_evaluated": reference is not None,
        "action_audit": "program_decides_before_actor; unchanged_deterministic_NN_command; physical_cancellations_separate"}
    if reference is not None:
        gates = {"candidate_gate": deepcopy(protocol["absolute_gate"]), "warmup_gate": deepcopy(protocol["warmup_gate"])}
        report["capability"] = capability(report, reference, random, gates)
        report["warmup_capability"] = warmup_capability(report, reference, gates, report["total_actor_training_joint_steps"])
        if allow_test_fixture:
            report["capability"]["eligible"] = report["warmup_capability"]["eligible"] = False
            report["capability"]["status"] = "test_fixture_not_a_candidate"
    else:
        report["capability"] = {"eligible": False, "status": "baselines_not_supplied_not_qualified"}
        report["warmup_capability"] = {"eligible": False, "checks": {}, "status": "baselines_not_supplied"}
    return report
