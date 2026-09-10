"""Explicitly authorized, paired continuations of one native foundation.

``--prepare`` checks existing evidence and copies states; it never samples or
optimizes. ``--run --additional-budget-authorized`` is a separate operation.
The additional budget is per-branch MAPPO joint transitions. Development and
extraction simulations are separately bounded, counted, and never presented
as optimizer transitions or as independent final-test evidence.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import tempfile
import time

import numpy as np
import torch

from env.warehouse_native.environment import NativeWarehouseEnv
from env.warehouse_native.feedback import FeedbackConfig, FeedbackManager
from env.warehouse_native.partners import partner_action
from env.warehouse_native.policy import ACTIONS, NumPyNativeActor
from env.warehouse_native.scenarios import reset_scenario
from .warehouse_native import NativeTrainer, acknowledge_update, atomic_torch_save, reserve_sampling
from .warehouse_native_common import ROOT, VERSION, atomic_json, append_jsonl, canonical, digest, file_hash, load_protocol, source_hashes
from .warehouse_native_evaluation import capability, critical_groups, evaluate, summarize


CONTINUATION_VERSION = "warehouse-native-paired-continuation.v1"
BRANCHES = ("control", "feedback")
DEVELOPMENT_PARTNERS = ("skilled", "assertive", "noisy")
EXTRACTION_PROFILES = ("selfplay", "skilled", "assertive", "noisy")
FORK_MUTABLE_FIELDS = {"feedback_enabled", "feedback_state", "acknowledgement", "continuation"}


def execution_sources():
    paths = [Path(__file__), ROOT / "env/warehouse_native/feedback.py"]
    paths += sorted((ROOT / "core").glob("*.py"))
    return {str(path.relative_to(ROOT)): file_hash(path) for path in paths}


def state_digest(value):
    """Canonical tensor-aware fingerprint, independent of torch.save zip headers."""
    result = sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            result.update(b"torch"); visit(array)
        elif isinstance(item, np.ndarray):
            result.update(b"array"); result.update(item.dtype.str.encode())
            result.update(str(item.shape).encode()); result.update(item.tobytes())
        elif isinstance(item, dict):
            result.update(b"dict")
            for key in sorted(item, key=lambda v: (type(v).__name__, str(v))):
                visit(key); visit(item[key])
        elif isinstance(item, (list, tuple)):
            result.update(type(item).__name__.encode()); result.update(str(len(item)).encode())
            for element in item:
                visit(element)
        else:
            result.update(type(item).__name__.encode()); result.update(canonical(item).encode())
    visit(value)
    return result.hexdigest()


def feedback_config(protocol):
    settings = protocol["feedback"]
    return FeedbackConfig(
        warmup_steps=protocol["warmup_gate"]["minimum_joint_steps"],
        ramp_steps=settings["ramp_steps"], lambda_max=settings["lambda_max"],
        minimum_fidelity=settings["minimum_fidelity"],
        minimum_critical_fidelity=settings["minimum_critical_fidelity"],
        maximum_mean_kl=settings["maximum_mean_kl"],
        maximum_performance_drop_fraction=settings["maximum_performance_drop_fraction"],
        depths=tuple(settings["depths"]), leaves=tuple(settings["leaves"]),
        critical_groups=tuple(protocol["explanation"]["categories"]), seed=protocol["seed"],
    )


def warmup_capability(report, reference, protocol, step):
    """Complete public-performance gate, including individual AI contribution."""
    gate = protocol["warmup_gate"]
    checks = {"minimum_joint_steps": step >= gate["minimum_joint_steps"]}
    for kind in DEVELOPMENT_PARTNERS:
        got, ref = report["summary"][kind], reference["summary"][kind]
        denominator = ref["mean_team_deliveries"]
        valid = denominator is not None and np.isfinite(denominator) and denominator > 0
        checks[kind + "_positive_reference"] = bool(valid)
        checks[kind + "_throughput"] = bool(valid and got["mean_team_deliveries"] / denominator >= gate["minimum_delivery_ratio_to_reference"])
        checks[kind + "_ai_contribution"] = got["mean_ai_deliveries"] >= gate["minimum_ai_deliveries_mean"]
        checks[kind + "_ai_survival"] = got["ai_active_end_rate"] >= gate["minimum_ai_active_end_rate"]
        checks[kind + "_sample_count"] = got["episodes"] >= protocol["candidate_gate"]["episodes_per_partner"]
    return {"eligible": all(checks.values()), "checks": checks}


def _verified_report(path, scenarios):
    report = json.loads(Path(path).read_text())
    expected = {scene["id"]: scene["fingerprint"] for scene in scenarios}
    for kind in DEVELOPMENT_PARTNERS:
        rows = [row for row in report["rows"] if row["partner"] == kind]
        if len(rows) != len(expected) or {row["scenario_id"] for row in rows} != set(expected):
            raise ValueError("Development report is not the exact frozen validation set")
        if any(row["initial_fingerprint"] != expected[row["scenario_id"]] or row["nn_action_overrides"] != 0 for row in rows):
            raise ValueError("Development evidence has a scene or neural-action mismatch")
        if canonical(summarize(rows)) != canonical(report["summary"][kind]):
            raise ValueError("Development summary does not match its recorded episodes")
    return report


def _same_actor(payload, actor):
    if actor.obs_dim != payload["obs_dim"] or actor.state_dim != payload["state_dim"]:
        raise ValueError("Foundation Actor dimensions differ from the checkpoint")
    for name, array in actor.weights.items():
        if not np.array_equal(payload["model"]["actor." + name].detach().cpu().numpy(), array):
            raise ValueError("Foundation validation Actor differs from its checkpoint")


def fork_payload(payload, branch, manager, *, pair_id, additional_steps):
    if branch not in BRANCHES or payload["feedback_enabled"]:
        raise ValueError("Only a pure-RL foundation can start this explicit pair")
    fork = copy.deepcopy(payload)
    fork.pop("acknowledgement", None)
    fork["feedback_enabled"] = branch == "feedback"
    fork["feedback_state"] = manager.state_dict() if branch == "feedback" else None
    fork["continuation"] = {
        "version": CONTINUATION_VERSION, "pair_id": pair_id, "branch": branch,
        "foundation_joint_steps": payload["joint_steps"], "additional_budget": additional_steps,
        "additional_joint_steps": 0, "initial_rng_offset": 0,
        "last_refresh_step": None, "latest_gate": None, "best_continuation": None,
        "pending_refresh_step": payload["joint_steps"],
        "auxiliary_actual_joint_steps": 0, "audit_events": [],
    }
    before = {key: value for key, value in payload.items() if key not in FORK_MUTABLE_FIELDS}
    after = {key: value for key, value in fork.items() if key not in FORK_MUTABLE_FIELDS}
    if state_digest(before) != state_digest(after):
        raise AssertionError("Explicit feedback fork changed neural/optimizer/environment/RNG state")
    return fork


def prepare_pair(foundation, checkpoint, output, additional_steps, *, expected_protocol=None):
    """Copy a reviewable pair only after validating already-existing evidence."""
    foundation, checkpoint, output = Path(foundation).resolve(), Path(checkpoint).resolve(), Path(output).resolve()
    if output == foundation or foundation in output.parents and output.name in {"checkpoints", "validation"}:
        raise ValueError("Continuation output must not replace foundation artifacts")
    if output.exists() and any(path.name != "prepare_report.json" for path in output.iterdir()):
        raise ValueError("Pair preparation requires a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    evidence = {"foundation_checkpoint": str(checkpoint)}
    try:
        protocol = load_protocol() if expected_protocol is None else copy.deepcopy(expected_protocol)
        n = protocol["training"]["environments"]
        if type(additional_steps) is not int or additional_steps <= 0 or additional_steps % n:
            raise ValueError("Each additional branch budget must be a positive multiple of environment count")
        if not checkpoint.is_relative_to(foundation / "checkpoints"):
            raise ValueError("Preparation requires an archived checkpoint inside the foundation directory")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        scenarios = json.loads((foundation / "scenarios.json").read_text())
        if payload["version"] != VERSION or payload["protocol"] != protocol or payload["sources"] != source_hashes():
            raise ValueError("Foundation source or protocol has changed")
        if payload["feedback_enabled"] or payload["feedback_state"] is not None:
            raise ValueError("Foundation must be pure RL without a previous feedback target")
        if payload["scenario_manifest_hash"] != digest(scenarios):
            raise ValueError("Foundation checkpoint does not bind this scenario manifest")
        step = payload["joint_steps"]
        evidence["foundation_joint_steps"] = step
        if step < protocol["warmup_gate"]["minimum_joint_steps"] or step % n:
            raise ValueError("Foundation checkpoint has not reached the registered warmup")
        if step > protocol["authorized_run"]["joint_steps_max"]:
            raise ValueError("Source exceeds the registered foundation budget")
        actor_path = foundation / f"checkpoints/actor_{step:07d}.npz"
        actor = NumPyNativeActor(actor_path)
        _same_actor(payload, actor)
        evidence["foundation_actor_sha256"] = actor.sha256
        if actor.metadata["joint_steps"] != step or actor.metadata.get("feedback_enabled") is not False:
            raise ValueError("Foundation Actor metadata is not the exact pure-RL checkpoint")
        report_path = foundation / f"validation/step_{step:07d}.json"
        report = _verified_report(report_path, scenarios["splits"]["validation"])
        reference = _verified_report(foundation / "validation_reference.json", scenarios["splits"]["validation"])
        random_report = _verified_report(foundation / "validation_random.json", scenarios["splits"]["validation"])
        if report.get("actor_sha256") != actor.sha256 or report.get("joint_steps") != step:
            raise ValueError("Foundation validation does not bind the selected Actor and step")
        gate = capability(report, reference, random_report, protocol)
        evidence.update(capability=gate, validation_summary=report["summary"],
                        reference_summary=reference["summary"], random_summary=random_report["summary"])
        if not gate["eligible"]:
            raise ValueError("Foundation capability gate failed; no continuation pair may be created")
        if len(scenarios["splits"]["train"]) < 2 or len(scenarios["splits"]["extraction"]) < 2:
            raise ValueError("Independent training and extraction-development pools are required")
        for split in ("train", "extraction", "validation"):
            fingerprints = [scene["fingerprint"] for scene in scenarios["splits"][split]]
            if len(fingerprints) != len(set(fingerprints)):
                raise ValueError("Duplicate physical scenario inside a development pool")
        sets = [{scene["fingerprint"] for scene in scenarios["splits"][split]} for split in ("train", "extraction", "validation")]
        if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
            raise ValueError("Training/extraction-development/validation physical scenarios overlap")
    except (ValueError, KeyError, OSError) as error:
        result = {"version": CONTINUATION_VERSION, "status": "blocked", "reason": str(error), **evidence,
                  "training_started": False, "environment_transitions": 0,
                  "additional_training_authorized": False}
        atomic_json(output / "prepare_report.json", result)
        return result
    interval = protocol["feedback"]["extract_interval"]
    refreshes = math.ceil(additional_steps / interval) + 1
    pool_count = min(32, len(scenarios["splits"]["train"]), len(scenarios["splits"]["extraction"]))
    horizon = int(scenarios["configuration"]["horizon"])
    per_refresh = horizon * (3 * len(scenarios["splits"]["validation"]) + 2 * pool_count)
    plan = {
        "version": CONTINUATION_VERSION, "foundation_checkpoint_sha256": file_hash(checkpoint),
        "foundation_actor_sha256": actor.sha256, "foundation_joint_steps": step,
        "additional_joint_steps_per_branch": additional_steps,
        "target_absolute_joint_steps": step + additional_steps,
        "training_sources": source_hashes(), "execution_sources": execution_sources(),
        "scenario_manifest_sha256": digest(scenarios), "protocol_sha256": digest(protocol),
        "capability": gate, "foundation_directory": str(foundation),
        "foundation_checkpoint": str(checkpoint), "paired_initial_rng_offset": {b: 0 for b in BRANCHES},
        "training_replay_rng_offset": {b: 0 for b in BRANCHES},
        "extraction": {"fit_pool": "train", "selection_pool": "extraction",
                       "scenarios_per_pool": pool_count, "profiles": list(EXTRACTION_PROFILES),
                       "selection_pool_role": "reused_development_not_final_test",
                       "sampling": "same_frozen_actor_stochastic_actions_and_soft_probabilities",
                       "refresh_interval": interval, "minimum_shared_observations_after_filter": False},
        "validation": {"pool": "validation", "role": "reused_development_model_selection",
                       "final_test_read": False, "explanation_test_read": False, "play_read": False},
        "auxiliary_sampling": {"separate_from_optimizer_joint_steps": True,
                               "actual_count_scope": "completed_checkpointed_audits_only",
                               "cap_per_branch": per_refresh * (refreshes + 2),
                               "planned_refreshes": refreshes, "crash_recovery_refresh_allowance": 2},
        "additional_training_authorized": False, "status": "prepared_not_authorized",
    }
    manager = FeedbackManager(actor.metadata["feature_names"], feedback_config(protocol))
    protected = {key: state_digest(value) for key, value in payload.items() if key not in FORK_MUTABLE_FIELDS}
    plan["preserved_checkpoint_fields"] = protected
    atomic_json(output / "protocol.json", protocol)
    atomic_json(output / "scenarios.json", scenarios)
    for source, destination in ((checkpoint, "foundation_checkpoint.pt"), (actor_path, "foundation_actor.npz"),
                                (report_path, "foundation_validation.json"),
                                (foundation / "validation_reference.json", "validation_reference.json"),
                                (foundation / "validation_random.json", "validation_random.json")):
        shutil.copyfile(source, output / destination)
    frozen = ("foundation_checkpoint.pt", "foundation_actor.npz", "foundation_validation.json",
              "validation_reference.json", "validation_random.json", "protocol.json", "scenarios.json")
    plan["frozen_files"] = {name: file_hash(output / name) for name in frozen}
    plan["pair_id"] = digest(plan)
    for branch in BRANCHES:
        folder = output / branch
        folder.mkdir()
        fork = fork_payload(payload, branch, manager, pair_id=plan["pair_id"], additional_steps=additional_steps)
        atomic_torch_save(folder / "initial_checkpoint.pt", fork)
        reserve_sampling(folder / "sampling_budget.json", 0, cap=additional_steps)
        reserve_sampling(folder / "auxiliary_budget.json", 0, cap=plan["auxiliary_sampling"]["cap_per_branch"])
    atomic_json(output / "pair.json", plan)
    result = {"version": CONTINUATION_VERSION, "status": "prepared_not_authorized", "pair_id": plan["pair_id"],
              "training_started": False, "environment_transitions": 0,
              "additional_training_authorized": False, "additional_joint_steps_per_branch": additional_steps}
    atomic_json(output / "prepare_report.json", result)
    return result


@contextmanager
def preserve_global_rng():
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    try:
        yield
    finally:
        random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])


def collect_neural_dataset(actor, scenarios, *, pool, refresh_step, seed, max_steps=None):
    """Only real executed neural-role rows; no program-role action labels."""
    if pool not in ("train", "extraction") or any(not scene["id"].startswith(pool + "_") for scene in scenarios):
        raise ValueError("Feedback trajectories may only use train or extraction-development scenes")
    observations, probabilities, episode_ids, groups, traces = [], [], [], [], []
    transitions = 0
    for index, scene in enumerate(scenarios):
        env = NativeWarehouseEnv(); reset_scenario(env, scene)
        profile = EXTRACTION_PROFILES[index % len(EXTRACTION_PROFILES)]
        program_role = -1 if profile == "selfplay" else index % 2
        actor_rng = np.random.default_rng(np.random.SeedSequence([seed, refresh_step, index, 91]))
        partner_rng = np.random.default_rng(np.random.SeedSequence([seed, refresh_step, index, 92]))
        decisions = []
        while not env.done and (max_steps is None or env.state.frame < max_steps):
            obs = env.observations()
            program = None if program_role < 0 else partner_action(env, f"robot_{program_role + 1}", profile, partner_rng)
            proposed, distribution = actor.act(obs, False, actor_rng)
            submitted = dict(proposed)
            if program_role >= 0:
                submitted[f"robot_{program_role + 1}"] = program
            row_indices = []
            for role, agent in enumerate(env.state.agents):
                if role == program_role or not agent.active:
                    continue
                assert proposed[agent.agent_id] == submitted[agent.agent_id]
                row_indices.append(len(observations))
                observations.append(obs[agent.agent_id]); probabilities.append(distribution[agent.agent_id])
                episode_ids.append(f"{pool}/{refresh_step}/{scene['id']}")
                groups.append(critical_groups(env, agent.agent_id))
            _, _, _, _, info = env.step(submitted)
            transitions += 1
            decisions.append({"frame": env.state.frame, "proposed": proposed, "submitted": submitted,
                              "executed": info["executed_actions"], "neural_row_indices": row_indices})
        traces.append({"scenario_id": scene["id"], "fingerprint": scene["fingerprint"],
                       "partner": profile, "program_role": program_role, "decisions": decisions})
    width = actor.obs_dim
    return {"observations": np.asarray(observations, dtype=np.float32).reshape(-1, width),
            "probabilities": np.asarray(probabilities, dtype=np.float32).reshape(-1, 5),
            "episode_ids": episode_ids, "groups": groups, "traces": traces,
            "joint_transitions": transitions, "actor_sha256": actor.sha256, "pool": pool,
            "teacher_rows_in_fit": 0, "neural_submitted_overrides": 0}


def disjoint_validation(train, validation):
    fingerprints = {row.tobytes() for row in train["observations"]}
    keep = np.asarray([row.tobytes() not in fingerprints for row in validation["observations"]])
    result = {**validation}
    for name in ("observations", "probabilities"):
        result[name] = validation[name][keep]
    for name in ("episode_ids", "groups"):
        result[name] = [value for value, present in zip(validation[name], keep) if present]
    result["exact_rows_removed_against_train"] = int((~keep).sum())
    return result


def _save_dataset(path, dataset):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, observations=dataset["observations"], probabilities=dataset["probabilities"],
                                episode_ids=np.asarray(dataset["episode_ids"]), groups_json=json.dumps(dataset["groups"]))
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def refresh_feedback(trainer, context, output, plan, scenarios, reference, random_report):
    """One frozen-Actor refresh; validation never enters the optimizer replay."""
    step = trainer.joint_steps
    folder = Path(output) / f"development/step_{step:07d}"
    folder.mkdir(parents=True, exist_ok=True)
    actor_path = trainer.export(folder / "actor.npz")
    actor = NumPyNativeActor(actor_path)
    n = plan["extraction"]["scenarios_per_pool"]
    horizon = scenarios["configuration"]["horizon"]
    upper = horizon * (len(scenarios["splits"]["validation"]) * len(DEVELOPMENT_PARTNERS)
                       + (2*n if trainer.feedback_enabled else 0))
    reserved = reserve_sampling(Path(output) / "auxiliary_budget.json", upper,
                                cap=plan["auxiliary_sampling"]["cap_per_branch"])
    started = time.perf_counter()
    with preserve_global_rng():
        report = evaluate(actor, scenarios["splits"]["validation"])
        actual = sum(row["steps"] for row in report["rows"])
        report.update(actor_sha256=actor.sha256, joint_steps=step,
                      capability=capability(report, reference, random_report, trainer.protocol),
                      dataset_role="repeated_development_validation_not_final_test")
        warmup = warmup_capability(report, reference, trainer.protocol, step)
        score = float(np.mean([report["summary"][p]["mean_team_deliveries"] for p in DEVELOPMENT_PARTNERS]))
        baseline = float(np.mean([reference["summary"][p]["mean_team_deliveries"] for p in DEVELOPMENT_PARTNERS]))
        fit_report = None
        if trainer.feedback_enabled:
            train = collect_neural_dataset(actor, scenarios["splits"]["train"][:n], pool="train",
                                           refresh_step=step, seed=trainer.seed)
            validation_raw = collect_neural_dataset(actor, scenarios["splits"]["extraction"][:n], pool="extraction",
                                                    refresh_step=step, seed=trainer.seed + 100)
            actual += train["joint_transitions"] + validation_raw["joint_transitions"]
            validation = disjoint_validation(train, validation_raw)
            _save_dataset(folder / "train_rows.npz", train)
            _save_dataset(folder / "selection_rows.npz", validation)
            atomic_json(folder / "trajectory_provenance.json", {
                "actor_sha256": actor.sha256, "train": train["traces"], "selection": validation_raw["traces"],
                "selection_pool": "extraction_reused_development", "final_test_read": False,
                "teacher_rows_in_fit": 0, "selection_overlap_rows_removed": validation["exact_rows_removed_against_train"],
                "selection_trace_row_indices_refer_to_unfiltered_pool": True,
            })
            try:
                fit_report = trainer.feedback.fit(
                    train["observations"], train["probabilities"], validation["observations"], validation["probabilities"],
                    step=step, source_actor_sha256=actor.sha256,
                    train_episode_ids=train["episode_ids"], val_episode_ids=validation["episode_ids"],
                    train_groups=train["groups"], val_groups=validation["groups"],
                )
                trainer.feedback.save_program(folder / "program.json")
            except (ValueError, RuntimeError) as error:
                # Failed extraction removes feedback; it never substitutes a
                # program action or prevents ordinary MAPPO from proceeding.
                trainer.feedback.current_lambda = 0.
                trainer.feedback.reliable = False
                fit_report = {"reliable": False, "reason": str(error), "actor_sha256": actor.sha256}
            gate = trainer.feedback.update(step, score, capability_eligible=warmup["eligible"], reference_score=baseline)
            atomic_json(folder / "fit.json", fit_report)
        else:
            gate = {"active": False, "lambda": 0., "reason": "pure_rl_control"}
    report["warmup_capability"] = warmup
    atomic_json(folder / "validation.json", report)
    event = {"step": step, "actor_sha256": actor.sha256, "validation_score": score,
             "reference_score": baseline, "warmup_eligible": warmup["eligible"], "gate": gate,
             "actual_auxiliary_joint_steps": actual, "reserved_auxiliary_upper_bound": reserved,
             "seconds": time.perf_counter()-started, "tree_fit": fit_report,
             "training_dataset_used": "train_only", "selection_data_reused_as_development": True}
    atomic_json(folder / "refresh.json", event)
    context["last_refresh_step"] = step
    context["pending_refresh_step"] = None
    context["latest_gate"] = {"validation_score": score, "reference_score": baseline,
                              "capability_eligible": warmup["eligible"]}
    context["auxiliary_actual_joint_steps"] += actual
    context["audit_events"].append({"step": step, "report": str((folder / "refresh.json").relative_to(output)),
                                   "sha256": file_hash(folder / "refresh.json")})
    shutdowns = float(np.mean([report["summary"][p]["mean_shutdowns"] for p in DEVELOPMENT_PARTNERS]))
    candidate = {"score": score, "shutdowns": shutdowns, "joint_steps": step,
                 "actor": str(actor_path.relative_to(output)), "actor_sha256": actor.sha256,
                 "capability": report["capability"]}
    best = context["best_continuation"]
    if best is None or (score, -shutdowns, -step) > (best["score"], -best["shutdowns"], -best["joint_steps"]):
        context["best_continuation"] = candidate
    return event


def _save_complete(output, trainer, context, acknowledgement=None):
    context["additional_joint_steps"] = trainer.joint_steps-context["foundation_joint_steps"]
    payload = trainer.state_dict()
    payload["continuation"] = copy.deepcopy(context)
    if acknowledgement is not None:
        payload["acknowledgement"] = acknowledgement
    atomic_torch_save(Path(output) / "latest_checkpoint.pt", payload)
    acknowledge_update(output, payload)
    return payload


def run_branch(pair, branch, *, additional_budget_authorized=False, device="cpu", stop_after_additional=None):
    """Resume the latest complete update, never replay an old acknowledged one."""
    if not additional_budget_authorized:
        raise PermissionError("Additional control/feedback training has not been authorized")
    if branch not in BRANCHES:
        raise ValueError("Unknown paired continuation branch")
    pair = Path(pair).resolve()
    plan = json.loads((pair / "pair.json").read_text())
    if digest({key: value for key, value in plan.items() if key != "pair_id"}) != plan["pair_id"]:
        raise ValueError("Prepared pair description changed")
    if plan["version"] != CONTINUATION_VERSION or plan["training_sources"] != source_hashes() or plan["execution_sources"] != execution_sources():
        raise ValueError("Prepared continuation source contract changed; do not reuse its checkpoints")
    expected_files = {"foundation_checkpoint.pt", "foundation_actor.npz", "foundation_validation.json",
                      "validation_reference.json", "validation_random.json", "protocol.json", "scenarios.json"}
    if set(plan["frozen_files"]) != expected_files or any(file_hash(pair/name) != expected for name, expected in plan["frozen_files"].items()):
        raise ValueError("Prepared foundation evidence changed")
    if file_hash(pair / "foundation_checkpoint.pt") != plan["foundation_checkpoint_sha256"]:
        raise ValueError("Copied foundation checkpoint changed")
    protocol = json.loads((pair / "protocol.json").read_text())
    scenarios = json.loads((pair / "scenarios.json").read_text())
    if digest(protocol) != plan["protocol_sha256"] or digest(scenarios) != plan["scenario_manifest_sha256"]:
        raise ValueError("Prepared protocol/scenes changed")
    output = pair / branch
    cap, base = plan["additional_joint_steps_per_branch"], plan["foundation_joint_steps"]
    n = protocol["training"]["environments"]
    stop = cap if stop_after_additional is None else stop_after_additional
    if type(stop) is not int or not 0 <= stop <= cap or stop % n:
        raise ValueError("Stop must be a valid additional-step boundary within the equal budget")
    locks = []
    try:
        for path in (pair / "pair.lock", Path(plan["foundation_directory"]) / "run.lock"):
            handle = path.open("a+")
            locks.append(handle)
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("The foundation or paired continuation is already running") from None
        execution = {"device": str(device), "torch_version": str(torch.__version__), "numpy_version": np.__version__}
        execution_path = pair / "execution.json"
        if execution_path.exists():
            if json.loads(execution_path.read_text()) != execution:
                raise ValueError("Paired branches and resumptions must use the same device and library versions")
        else:
            atomic_json(execution_path, execution)
        atomic_json(output / "authorization.json", {"version": CONTINUATION_VERSION,
            "additional_budget_authorized": True, "per_branch_additional_budget": cap,
            "auxiliary_budget_cap": plan["auxiliary_sampling"]["cap_per_branch"],
            "pid": os.getpid(), "branch": branch, "pair_id": plan["pair_id"]})
        reference = json.loads((pair / "validation_reference.json").read_text())
        random_report = json.loads((pair / "validation_random.json").read_text())
        initial = torch.load(output / "initial_checkpoint.pt", map_location="cpu", weights_only=False)
        check_fields = {key: state_digest(initial[key]) for key in plan["preserved_checkpoint_fields"]}
        if check_fields != plan["preserved_checkpoint_fields"]:
            raise ValueError("Paired initial checkpoint no longer matches the common foundation")
        path = output / "latest_checkpoint.pt"
        payload = torch.load(path, map_location=device, weights_only=False) if path.exists() else initial
        context = copy.deepcopy(payload["continuation"])
        if (context["pair_id"] != plan["pair_id"] or context["branch"] != branch or context["additional_budget"] != cap
                or context["foundation_joint_steps"] != base or context["additional_joint_steps"] != payload["joint_steps"]-base):
            raise ValueError("Continuation checkpoint counters or pair identity changed")
        log = output / "training.jsonl"
        if log.exists() and any(json.loads(line)["joint_steps"] > payload["joint_steps"] for line in log.read_text().splitlines() if line.strip()):
            raise ValueError("A newer acknowledged update exists; refusing to replay transitions")
        manager = FeedbackManager(NumPyNativeActor(pair / "foundation_actor.npz").metadata["feature_names"], feedback_config(protocol)) if branch == "feedback" else None
        trainer = NativeTrainer(protocol, scenarios, device=device, feedback=manager)
        trainer.load_state_dict(payload)
        acknowledge_update(output, payload)
        reserved = json.loads((output / "sampling_budget.json").read_text())["reserved_joint_steps"]
        completed = trainer.joint_steps-base
        if reserved < completed:
            raise ValueError("Sampling ledger is older than the latest complete update")
        discarded = reserved-completed
        target = base + min(stop, completed+cap-reserved)
        stop_requested = False
        def request_stop(signum, frame):
            nonlocal stop_requested
            stop_requested = True
        old_signals = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            if context["pending_refresh_step"] is not None:
                if context["pending_refresh_step"] != trainer.joint_steps:
                    raise ValueError("Pending development audit is bound to another checkpoint")
                refresh_feedback(trainer, context, output, plan, scenarios, reference, random_report)
                _save_complete(output, trainer, context)
            if trainer.joint_steps >= target:
                return {"status": "budget_complete" if completed == cap else "no_available_sampling_budget",
                        "joint_steps": trainer.joint_steps, "additional_joint_steps": completed,
                        "reserved_additional_upper_bound": reserved, "discarded_after_crash_upper_bound": discarded}
            while trainer.joint_steps < target:
                interval = protocol["feedback"]["extract_interval"]
                next_refresh = base + ((trainer.joint_steps-base)//interval+1)*interval
                boundary = min(target, next_refresh)
                length = min(protocol["training"]["rollout_steps"], (boundary-trainer.joint_steps)//n)
                if length <= 0:
                    raise ValueError("Refresh interval is incompatible with the environment batch")
                if trainer.feedback_enabled:
                    gate = context["latest_gate"]
                    trainer.feedback.update(trainer.joint_steps, gate["validation_score"],
                                            capability_eligible=gate["capability_eligible"], reference_score=gate["reference_score"])
                tick = time.perf_counter()
                reserved = reserve_sampling(output / "sampling_budget.json", length*n, cap=cap)
                batch = trainer.collect(length)
                collected = time.perf_counter()
                metrics = trainer.update(batch)
                elapsed = time.perf_counter()-tick
                trainer.elapsed_seconds += elapsed
                record = {"joint_steps": trainer.joint_steps, "foundation_joint_steps": base,
                    "additional_joint_steps": trainer.joint_steps-base, "branch": branch,
                    "optimizer_updates": trainer.optimizer_updates, "collection_seconds": collected-tick,
                    "update_seconds": elapsed-(collected-tick), "metrics": metrics, "execution_audit": batch["audit"],
                    "reserved_additional_upper_bound": reserved, "discarded_after_crash_upper_bound": discarded,
                    "feedback_lambda": trainer.feedback.current_lambda if trainer.feedback_enabled else 0.}
                if trainer.joint_steps == boundary:
                    context["pending_refresh_step"] = trainer.joint_steps
                checkpoint = _save_complete(output, trainer, context,
                    {"record": record, "episodes": list(trainer.completed_episodes)})
                trainer.completed_episodes.clear()
                if trainer.joint_steps == boundary:
                    # Commit the optimizer checkpoint before any expensive
                    # external audit, then archive the post-audit feedback state.
                    refresh_feedback(trainer, context, output, plan, scenarios, reference, random_report)
                    checkpoint = _save_complete(output, trainer, context)
                    atomic_torch_save(output / f"checkpoints/checkpoint_{trainer.joint_steps:07d}.pt", checkpoint)
                result = {"version": CONTINUATION_VERSION, "status": "running", "branch": branch,
                    "joint_steps": trainer.joint_steps, "additional_joint_steps": trainer.joint_steps-base,
                    "additional_budget": cap, "auxiliary_actual_joint_steps": context["auxiliary_actual_joint_steps"],
                    "latest_gate": context["latest_gate"], "best_continuation": context["best_continuation"],
                    "reserved_additional_upper_bound": reserved, "discarded_after_crash_upper_bound": discarded}
                atomic_json(output / "progress.json", result)
                if stop_requested:
                    result["status"] = "paused_after_complete_update"
                    atomic_json(output / "progress.json", result)
                    return result
            result["status"] = "budget_complete" if trainer.joint_steps-base == cap else "paused_or_sampling_reservations_exhausted"
            atomic_json(output / "progress.json", result)
            return result
        finally:
            for sig, previous in old_signals.items():
                signal.signal(sig, previous)
    finally:
        for handle in reversed(locks):
            handle.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--foundation", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--additional-steps", type=int)
    parser.add_argument("--branch", choices=(*BRANCHES, "both"), default="both")
    parser.add_argument("--additional-budget-authorized", action="store_true")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--stop-after-additional", type=int)
    args = parser.parse_args(argv)
    if args.prepare:
        if args.foundation is None or args.checkpoint is None or args.additional_steps is None:
            parser.error("Preparation requires foundation, archived checkpoint, and equal additional steps")
        report = prepare_pair(args.foundation, args.checkpoint, args.output, args.additional_steps)
        print(canonical(report), flush=True)
        return 0 if report["status"] == "prepared_not_authorized" else 2
    if not args.additional_budget_authorized:
        parser.error("No additional training is authorized; --prepare does not grant --run permission")
    if args.additional_steps is not None or args.foundation is not None or args.checkpoint is not None:
        parser.error("Run uses the frozen prepared pair; budgets and source cannot be changed")
    torch.set_num_threads(1)
    for branch in BRANCHES if args.branch == "both" else (args.branch,):
        report = run_branch(args.output, branch, additional_budget_authorized=True,
                            device=args.device, stop_after_additional=args.stop_after_additional)
        print(canonical(report), flush=True)
        if report["status"] == "paused_after_complete_update":
            return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
