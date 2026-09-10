"""Prepared foundation revision; explicit protocol approval precedes sampling.

The v2 checkpoint container and PPO are reused, but the experimental identity,
reward, curriculum selector, source closure, and budget are independently bound.
No existing training file, model, public physics, or runtime controller is edited.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import fcntl
import json
from pathlib import Path
import random
import shutil
import signal
import time

import numpy as np
import torch

from env.warehouse_native.policy import NativeActorCritic, NumPyNativeActor
from .warehouse_native import atomic_torch_save, acknowledge_update
from .warehouse_native_common import ROOT, atomic_json, canonical, digest, file_hash
from .warehouse_native_v2 import (
    NativeV2Trainer, IndependentOptimizers, VERSION as CONTAINER_VERSION,
    v2_source_hashes, evaluate_v2, diagnostic_review, checkpoint_selection_due,
)
from .warehouse_native_evaluation import capability
from .warehouse_native_diverse_curriculum import DiverseCurriculumBuilder, validate_diverse_curriculum_bank
from .warehouse_native_revision_reward import RevisedRewardEnvironment, REWARD_REVISION
from .warehouse_native_revision_budget import RevisionBudget

VERSION = "warehouse-native-foundation.r1"
PROTOCOL_PATH = Path(__file__).with_name("warehouse_native_revision_protocol.json")


def revision_source_hashes():
    result = v2_source_hashes()
    for name in ("warehouse_native_revision.py", "warehouse_native_revision_protocol.json",
                 "warehouse_native_diverse_curriculum.py", "warehouse_native_revision_reward.py",
                 "warehouse_native_revision_budget.py"):
        path = Path(__file__).with_name(name)
        result[str(path.relative_to(ROOT))] = file_hash(path)
    return result


def load_revision_protocol():
    protocol = json.loads(PROTOCOL_PATH.read_text())
    validate_revision_protocol(protocol)
    return protocol


def validate_revision_protocol(protocol):
    revision = protocol.get("experiment_revision", {})
    if protocol.get("version") != CONTAINER_VERSION or revision.get("version") != VERSION:
        raise ValueError("Revision identity and checkpoint container differ")
    if not revision.get("fresh_initialization") or not revision.get("public_physics_and_score_unchanged"):
        raise ValueError("Revision requires fresh initialization and unchanged public rules")
    if protocol["architecture"]["actions"] != ["UP", "DOWN", "LEFT", "RIGHT", "WAIT"]:
        raise ValueError("All five native actions must remain available")
    if protocol["architecture"]["masks"] or protocol["architecture"]["runtime_override"]:
        raise ValueError("Revision cannot add runtime control")
    if not 0 < revision["maximum_remaining_training_related_steps"] <= 290000:
        raise ValueError("Revision cannot expand the remaining budget")
    if protocol["authorized_run"]["feedback_enabled"]:
        raise ValueError("This revision is foundation-only")
    if protocol["reward"]["gamma"] != protocol["training"]["gamma"]:
        raise ValueError("Potential and PPO discount mismatch")
    return protocol


def validate_revision_bank(bank, scenarios, protocol):
    if (bank.get("protocol_sha256") != digest(protocol)
            or bank.get("experiment_revision") != protocol["experiment_revision"]
            or bank.get("selection_config") != protocol["diverse_curriculum"]
            or bank.get("entries_per_category") != protocol["curriculum"]["entries_per_category"]):
        raise ValueError("Curriculum belongs to another revision protocol")
    return validate_diverse_curriculum_bank(bank, scenarios, replay=False)


class NativeRevisionTrainer(NativeV2Trainer):
    """Same PPO, with explicit environment construction and revised provenance."""

    def __init__(self, protocol, scenarios, bank, *, device="cpu"):
        validate_revision_protocol(protocol)
        self.protocol, self.scenarios, self.bank = map(copy.deepcopy, (protocol, scenarios, bank))
        self.curriculum_coverage = validate_revision_bank(bank, scenarios, protocol)
        self.cfg = self.protocol["training"]
        self.device = torch.device(device)
        self.seed = self.protocol["seed"]
        random.seed(self.seed)
        np.random.seed(self.seed)
        # Avoid initializing MPS merely to run a CPU correctness fixture.
        torch.random.default_generator.manual_seed(self.seed)
        if self.device.type == "mps":
            torch.mps.manual_seed(self.seed)
        self.rng = np.random.default_rng(np.random.SeedSequence([self.seed, 401]))
        self.curriculum_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 402]))
        self.envs = [RevisedRewardEnvironment(
            reward_config=protocol["reward"],
            collision_cost=protocol["experiment_revision"]["collision_training_cost"])
            for _ in range(self.cfg["environments"])]
        count = len(self.envs)
        self.partner_kinds = [""] * count
        self.program_roles = [-1] * count
        self.scenario_ids = [""] * count
        self.episode_context = [None] * count
        self.episode_returns = np.zeros(count, dtype=np.float64)
        self.episode_reward_components = [Counter() for _ in range(count)]
        self.completed_episodes = []
        self.episode_count = self.joint_steps = self.optimizer_updates = self.minibatch_updates = 0
        self.best = None
        self.last_evaluated_joint_steps = 0
        self.elapsed_seconds = 0.
        self.feedback_enabled = False
        for i in range(count):
            self.reset_one(i)
        self.model = NativeActorCritic(self.envs[0].observation_size, len(self.envs[0].global_state())).to(self.device)
        self.optimizers = IndependentOptimizers(self.model, self.cfg)

    def state_dict(self):
        result = super().state_dict()
        result.update(experiment_revision=VERSION, revision_sources=revision_source_hashes(),
                      reward_revision=REWARD_REVISION,
                      last_evaluated_joint_steps=self.last_evaluated_joint_steps,
                      training_device=self.device.type)
        return result

    def load_state_dict(self, payload):
        if (payload.get("experiment_revision") != VERSION
                or payload.get("revision_sources") != revision_source_hashes()
                or payload.get("reward_revision") != REWARD_REVISION):
            raise ValueError("Cannot resume another experiment as this revision")
        if (payload.get("training_device") != self.device.type
                or (self.device.type == "mps" and "mps_rng" not in payload)):
            raise ValueError("Revision resume device or RNG state changed")
        evaluated = payload.get("last_evaluated_joint_steps")
        if type(evaluated) is not int or not 0 <= evaluated <= payload["joint_steps"]:
            raise ValueError("Invalid persisted evaluation boundary")
        if any(type(env) is not RevisedRewardEnvironment for env in self.envs):
            raise ValueError("Revision requires its exact revised reward environment")
        super().load_state_dict(payload)
        self.last_evaluated_joint_steps = evaluated

    def export(self, path):
        return self.model.export_npz(path, {
            "experiment_version": VERSION, "checkpoint_container_version": CONTAINER_VERSION,
            "joint_steps": self.joint_steps, "curriculum_generation_steps": self.bank["generation_steps"],
            "optimizer_updates": self.optimizer_updates, "protocol_sha256": digest(self.protocol),
            "scenario_manifest_sha256": digest(self.scenarios), "curriculum_bank_sha256": digest(self.bank),
            "source_sha256": digest(revision_source_hashes()), "initialization": "fresh_random",
            "candidate": True, "feedback_enabled": False, "feature_names": self.envs[0].feature_names,
            "reward_revision": REWARD_REVISION, "parent_model_loaded": False,
        })


def inspect_parent(parent, protocol):
    """Read existing authoritative records only; never loads old model weights."""
    parent = Path(parent).absolute()
    if parent != parent.resolve():
        raise ValueError("Parent path may not contain symbolic aliases")
    record = json.loads((parent / "run.json").read_text())
    progress = json.loads((parent / "progress.json").read_text())
    budget = json.loads((parent / "sampling_budget.json").read_text())
    scenarios = json.loads((parent / "scenarios.json").read_text())
    parent_protocol = json.loads((parent / "protocol.json").read_text())
    if record["version"] != CONTAINER_VERSION or record["sources"] != v2_source_hashes():
        raise ValueError("Parent is not the preserved frozen v2 run")
    if record["scenario_sha256"] != digest(scenarios):
        raise ValueError("Parent scene manifest changed")
    if record["protocol_sha256"] != digest(parent_protocol):
        raise ValueError("Parent protocol changed")
    baselines = {}
    expected_scenes = {row["id"]: row["fingerprint"] for row in scenarios["splits"]["validation"]}
    for kind in ("reference", "random"):
        path = parent / f"validation_{kind}.json"
        report = json.loads(path.read_text())
        for partner in protocol["checkpoint_selection"]["partners"]:
            rows = [row for row in report["rows"] if row["partner"] == partner]
            if (len(rows) != len(expected_scenes)
                    or {row["scenario_id"]: row["initial_fingerprint"] for row in rows} != expected_scenes):
                raise ValueError("Parent validation baseline scene binding differs")
        baselines[path.name] = file_hash(path)
    consumed = sum(budget["reserved"].values())
    if (budget["cap"] != 500000 or consumed != protocol["experiment_revision"]["parent_consumed_training_related_steps"]
            or progress["status"] != "v2_paused_for_diagnostic_review"):
        raise ValueError("Parent budget or diagnostic pause changed; reassess remaining allocation")
    return {"parent_run": str(parent), "reserved_before_revision": consumed,
            "global_cap": budget["cap"], "remaining_before_revision": budget["cap"] - consumed,
            "parent_record_sha256": file_hash(parent / "run.json"),
            "parent_protocol_sha256": file_hash(parent / "protocol.json"),
            "parent_budget_sha256": file_hash(parent / "sampling_budget.json"),
            "parent_progress_sha256": file_hash(parent / "progress.json"),
            "validation_baseline_sha256": baselines}, scenarios


def prepare(parent, output):
    protocol = load_revision_protocol()
    binding, scenarios = inspect_parent(parent, protocol)
    budget_preview = RevisionBudget(parent, "r1_prepare_preview",
        requested_cap=protocol["experiment_revision"]["maximum_remaining_training_related_steps"]).read()
    output = Path(output).absolute()
    if output != output.resolve() or Path(parent).resolve() in output.parents or output == Path(parent).resolve():
        raise ValueError("Preparation must be separate from the parent run")
    if output.exists():
        raise ValueError("Preparation requires a fresh output directory")
    sources = revision_source_hashes()
    output.mkdir(parents=True)
    atomic_json(output / "proposal.json", {
        "version": VERSION, "status": "prepared_requires_protocol_revision_approval",
        "protocol_sha256": digest(protocol), "sources": sources, "parent": binding,
        "maximum_training_related_steps": min(binding["remaining_before_revision"], budget_preview["global_remaining"],
            protocol["experiment_revision"]["maximum_remaining_training_related_steps"]),
        "budget_preview": budget_preview,
        "curriculum_steps_max": protocol["curriculum"]["maximum_generation_steps"],
        "new_training_authorized": False, "environment_steps": 0, "neural_updates": 0,
        "old_v2_ledger_modified": False, "feedback_training_included": False,
        "validation_scope": protocol["experiment_revision"]["validation_scope"],
        "release_bridge_status": "Unregistered experimental revision; full release provenance acceptance still required.",
    })
    atomic_json(output / "protocol.json", protocol)
    # Reusing a development split is disclosed, not claimed to be a fresh test.
    atomic_json(output / "scenarios.json", scenarios)
    for relative, expected in sources.items():
        destination = output / "source_bundle" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
        if file_hash(destination) != expected:
            raise ValueError("Source changed during preparation")
    return json.loads((output / "proposal.json").read_text())


def run(parent, output, *, revision_authorized=False, device="cpu", resume=False,
        stop_after_ppo_steps=None, resume_after_diagnostic_review=False):
    if not revision_authorized:
        raise ValueError("Explicit approval to change the frozen training protocol is required before sampling")
    protocol = load_revision_protocol()
    binding, scenarios = inspect_parent(parent, protocol)
    output = Path(output).absolute()
    parent = Path(parent).resolve()
    if output != output.resolve() or output == parent or parent in output.parents:
        raise ValueError("Revision output must be independent of the parent")
    n = protocol["training"]["environments"]
    # Only an intact rollout boundary may be used for the initial throughput probe.
    if stop_after_ppo_steps is not None and (type(stop_after_ppo_steps) is not int
            or not 0 < stop_after_ppo_steps < protocol["training"]["checkpoint_interval"]
            or stop_after_ppo_steps % (n * protocol["training"]["rollout_steps"])):
        raise ValueError("Initial probe must end on a full rollout before the first validation boundary")
    if device not in ("cpu", "mps"):
        raise ValueError("Unsupported training device")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    child_id = "r1_" + digest({"output": str(output), "protocol": digest(protocol)})[:24]
    if resume:
        record = json.loads((output / "run.json").read_text())
        if (record["version"] != VERSION or record["sources"] != revision_source_hashes()
                or record["protocol_sha256"] != digest(protocol) or record["parent"] != binding
                or record["device"] != device or record["child_id"] != child_id
                or record.get("approved_child_cap") != protocol["experiment_revision"]["maximum_remaining_training_related_steps"]
                or record.get("explicit_revision_flag_recorded") is not True
                or record.get("feedback_enabled") is not False or record.get("parent_model_loaded") is not False
                or json.loads((output / "protocol.json").read_text()) != protocol
                or json.loads((output / "scenarios.json").read_text()) != scenarios):
            raise ValueError("Revision resume provenance changed")
        for relative, expected in record["sources"].items():
            if file_hash(output / "source_bundle" / relative) != expected:
                raise ValueError("Revision source archive changed")
    elif output.exists():
        raise ValueError("A new revision needs a fresh output directory")
    budget = RevisionBudget(parent, child_id,
        requested_cap=protocol["experiment_revision"]["maximum_remaining_training_related_steps"],
        revision_authorized=True)
    torch.set_num_threads(1)
    with budget.lease():
        if inspect_parent(parent, protocol)[0] != binding:
            raise ValueError("Parent changed before acquiring the training lease")
        output.mkdir(parents=True, exist_ok=resume)
        run_lock = (output / "run.lock").open("a")
        try:
            fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not resume:
                atomic_json(output / "protocol.json", protocol)
                atomic_json(output / "scenarios.json", scenarios)
                atomic_json(output / "run.json", {"version": VERSION, "status": "authorized_revision_started",
                    "sources": revision_source_hashes(), "protocol_sha256": digest(protocol), "parent": binding,
                    "child_id": child_id, "approved_child_cap": protocol["experiment_revision"]["maximum_remaining_training_related_steps"],
                    "explicit_revision_flag_recorded": True,
                    "device": device, "feedback_enabled": False, "parent_model_loaded": False})
                for relative, expected in revision_source_hashes().items():
                    target = output / "source_bundle" / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT / relative, target)
                    if file_hash(target) != expected:
                        raise ValueError("Training source changed during archiving")
            return _sample_locked(parent, output, protocol, scenarios, budget, device,
                                  stop_after_ppo_steps, resume_after_diagnostic_review)
        finally:
            run_lock.close()


def _run_binding(output, budget):
    record = json.loads((output / "run.json").read_text())
    return {"run_record_sha256": digest(record), "child_id": record["child_id"],
            "parent": record["parent"], "registry_path": str(budget.registry_path)}


def _bound_checkpoint(trainer, output, budget):
    payload = trainer.state_dict()
    payload.update(run_binding=_run_binding(output, budget), budget_at_checkpoint=budget.read())
    return payload


def _check_persisted_budget(payload, output, budget, *, ppo_steps=0, curriculum_steps=0):
    current = budget.read()
    old = payload.get("budget_at_checkpoint", {})
    if payload.get("run_binding") != _run_binding(output, budget):
        raise ValueError("Persisted state belongs to another training run")
    for key in ("parent_run", "parent_sampling_budget_sha256", "parent_baseline", "child_id", "registry_path"):
        if old.get(key) != current.get(key):
            raise ValueError("Persisted budget lineage changed")
    reserved = old.get("child_reserved", {})
    for kind, actual in (("curriculum", curriculum_steps), ("ppo", ppo_steps)):
        if (type(reserved.get(kind)) is not int or not 0 <= actual <= reserved[kind]
                or reserved[kind] > current["child_reserved"][kind]):
            raise ValueError("Persisted transitions exceed their budget reservation")


def _finalize_boundary(trainer, output, protocol, scenarios, budget, reference, random_report):
    """Complete interrupted selection/review before allowing another PPO step."""
    interval, n = protocol["training"]["checkpoint_interval"], protocol["training"]["environments"]
    if not trainer.joint_steps or not checkpoint_selection_due(trainer.joint_steps, budget.remaining, n, interval):
        return False
    step = trainer.joint_steps
    actor_path = output / f"checkpoints/actor_{step:07d}.npz"
    report_path = output / f"validation/step_{step:07d}.json"
    if report_path.exists():
        actor = NumPyNativeActor(actor_path)
        report = json.loads(report_path.read_text())
        if (report.get("actor_sha256") != actor.artifact_sha256 or report.get("joint_steps") != step
                or report.get("run_binding") != _run_binding(output, budget)
                or report.get("experiment_revision") != VERSION):
            raise ValueError("Saved validation has a different Actor or run binding")
        for name, value in trainer.model.actor.state_dict().items():
            if not np.array_equal(actor.weights[name], value.detach().cpu().numpy()):
                raise ValueError("Validation Actor differs from resumed checkpoint")
    else:
        trainer.export(actor_path)
        actor = NumPyNativeActor(actor_path)
        report = evaluate_v2(actor, scenarios["splits"]["validation"], trainer.envs[0].feature_names)
        report["capability"] = capability(report, reference, random_report, protocol)
        report.update(joint_steps=step, actor_sha256=actor.artifact_sha256,
            experiment_revision=VERSION, run_binding=_run_binding(output, budget),
            validation_scope=protocol["experiment_revision"]["validation_scope"])
        atomic_json(report_path, report)
    score = float(np.mean([v["mean_team_deliveries"] for v in report["summary"].values()]))
    shutdowns = float(np.mean([v["mean_shutdowns"] for v in report["summary"].values()]))
    candidate = {"score": score, "shutdowns": shutdowns, "joint_steps": step,
        "actor": str(actor_path.relative_to(output)), "actor_sha256": actor.artifact_sha256,
        "capability": report["capability"]}
    if trainer.best is None or (score, -shutdowns, -step) > (trainer.best["score"], -trainer.best["shutdowns"], -trainer.best["joint_steps"]):
        trainer.best = candidate
    review = diagnostic_review(report, step, protocol)
    atomic_json(output / f"diagnostics/step_{step:07d}.json", review)
    if review["pause_required"]:
        atomic_json(output / "diagnostic_review_required.json", review)
    trainer.last_evaluated_joint_steps = step
    payload = _bound_checkpoint(trainer, output, budget)
    atomic_torch_save(output / f"checkpoints/checkpoint_{step:07d}.pt", payload)
    atomic_torch_save(output / "latest_checkpoint.pt", payload)
    ack = output / "diagnostic_review_acknowledged.json"
    acknowledged = ack.exists() and json.loads(ack.read_text()).get("review_sha256") == digest(review)
    pause = review["pause_required"] and not acknowledged
    atomic_json(output / "progress.json", {"status": "revision_paused_for_diagnostic_review" if pause else "revision_candidate_training",
        "joint_steps": step, "budget": budget.read(), "best": trainer.best, "review": review, "formal_ready": False})
    print(canonical({"event": "revision_checkpoint", "joint_steps": step,
        "budget": budget.read(), "capability": report["capability"], "pause_required": pause}), flush=True)
    return pause


def _sample_locked(parent, output, protocol, scenarios, budget, device, stop_after, review_ack):
    stopped = False
    def stop_signal(*_):
        nonlocal stopped
        stopped = True
    old_handlers = {s: signal.signal(s, stop_signal) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        review_path = output / "diagnostic_review_required.json"
        if review_path.exists():
            review = json.loads(review_path.read_text())
            if review["pause_required"] and not review_ack:
                raise ValueError("Revision is paused for diagnostic review")
            if review["pause_required"]:
                atomic_json(output / "diagnostic_review_acknowledged.json", {
                    "review_sha256": digest(review), "explicit_review_flag": True, "budget": budget.read()})
        bank_path = output / "curriculum_bank.json"
        if not bank_path.exists():
            builder = DiverseCurriculumBuilder(protocol, scenarios)
            builder_path = output / "curriculum_generation_latest.json"
            if builder_path.exists():
                saved = json.loads(builder_path.read_text())
                _check_persisted_budget(saved, output, budget, curriculum_steps=saved["builder"]["steps"])
                builder.load_state_dict(saved["builder"])
            maximum = protocol["curriculum"]["maximum_generation_steps"]
            while budget.child_reserved["curriculum"] < maximum and budget.remaining:
                count = min(protocol["curriculum"]["generation_chunk"], maximum - budget.child_reserved["curriculum"], budget.remaining)
                budget.reserve("curriculum", count)
                builder.advance(count)
                atomic_json(builder_path, {"builder": builder.state_dict(), "run_binding": _run_binding(output, budget),
                    "budget_at_checkpoint": budget.read()})
                if stopped:
                    return 130
            bank = builder.bank()
            atomic_json(output / "curriculum_binding.json", {"bank_sha256": digest(bank),
                "run_binding": _run_binding(output, budget), "budget_at_checkpoint": budget.read()})
            atomic_json(bank_path, bank)
        bank = json.loads(bank_path.read_text())
        binding = json.loads((output / "curriculum_binding.json").read_text())
        _check_persisted_budget(binding, output, budget, curriculum_steps=bank["generation_steps"])
        if binding["bank_sha256"] != digest(bank):
            raise ValueError("Final curriculum binding changed")
        try:
            coverage = validate_revision_bank(bank, scenarios, protocol)
        except ValueError as error:
            atomic_json(output / "progress.json", {"status": "revision_curriculum_quality_failed",
                "error": str(error), "budget": budget.read(), "joint_steps": 0,
                "candidate": True, "formal_ready": False, "neural_updates": 0,
                "coverage": getattr(error, "coverage", None)})
            return 0
        atomic_json(output / "curriculum_quality.json", coverage)
        latest = output / "latest_checkpoint.pt"
        if not latest.exists() and (budget.child_reserved["ppo"] or (output / "initial_actor.npz").exists()):
            raise ValueError("Missing checkpoint cannot silently restart a previously initialized run")
        trainer = NativeRevisionTrainer(protocol, scenarios, bank, device=device)
        if latest.exists():
            payload = torch.load(latest, map_location=device, weights_only=False)
            _check_persisted_budget(payload, output, budget,
                ppo_steps=payload["joint_steps"], curriculum_steps=bank["generation_steps"])
            trainer.load_state_dict(payload)
            acknowledge_update(output, payload)
        else:
            trainer.export(output / "initial_actor.npz")
            atomic_torch_save(latest, _bound_checkpoint(trainer, output, budget))
        if (budget.child_reserved["ppo"] < trainer.joint_steps
                or budget.child_reserved["curriculum"] < bank["generation_steps"]):
            raise ValueError("Revision budget is older than saved transitions")
        for kind in ("reference", "random"):
            src = parent / f"validation_{kind}.json"
            dst = output / src.name
            if not dst.exists():
                shutil.copyfile(src, dst)
            if file_hash(src) != file_hash(dst):
                raise ValueError("Reused physical validation baseline changed")
        reference = json.loads((output / "validation_reference.json").read_text())
        random_report = json.loads((output / "validation_random.json").read_text())
        if _finalize_boundary(trainer, output, protocol, scenarios, budget, reference, random_report):
            return 0
        n = protocol["training"]["environments"]
        interval = protocol["training"]["checkpoint_interval"]
        while budget.remaining >= n:
            if stop_after is not None and trainer.joint_steps >= stop_after:
                break
            boundary = (trainer.joint_steps // interval + 1) * interval
            limit = min(boundary, trainer.joint_steps + budget.remaining // n * n, stop_after or 10**12)
            count = min(protocol["training"]["rollout_steps"], (limit - trainer.joint_steps) // n)
            if count <= 0:
                break
            budget.reserve("ppo", count * n)
            start = time.perf_counter()
            batch = trainer.collect(count)
            metrics = trainer.update(batch)
            trainer.elapsed_seconds += time.perf_counter() - start
            record = {"version": VERSION, "joint_steps": trainer.joint_steps,
                "curriculum_generation_steps": bank["generation_steps"], "optimizer_updates": trainer.optimizer_updates,
                "metrics": metrics, "execution_audit": batch["audit"], "budget": budget.read()}
            payload = _bound_checkpoint(trainer, output, budget)
            payload["acknowledgement"] = {"record": record, "episodes": list(trainer.completed_episodes)}
            atomic_torch_save(latest, payload)
            acknowledge_update(output, payload)
            trainer.completed_episodes.clear()
            at_limit = trainer.joint_steps == limit
            selected = checkpoint_selection_due(trainer.joint_steps, budget.remaining, n, interval)
            if at_limit and not selected:
                trainer.export(output / f"diagnostics/probe_actor_{trainer.joint_steps:07d}.npz")
                print(canonical({"event": "revision_probe", "ppo_joint_steps": trainer.joint_steps,
                    "budget": budget.read(), "ppo_steps_per_training_second": trainer.joint_steps / max(trainer.elapsed_seconds, 1e-9),
                    "throughput_excludes_curriculum_and_validation": True, "used_for_selection": False}), flush=True)
            if at_limit and selected:
                if _finalize_boundary(trainer, output, protocol, scenarios, budget, reference, random_report):
                    return 0
            if stopped:
                return 130
        atomic_json(output / "progress.json", {"status": "revision_paused" if budget.remaining >= n else "revision_budget_complete_candidate",
            "joint_steps": trainer.joint_steps, "curriculum_generation_steps": bank["generation_steps"],
            "budget": budget.read(), "best": trainer.best, "formal_ready": False})
        return 0
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision-authorized", action="store_true")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-ppo-steps", type=int)
    parser.add_argument("--resume-after-diagnostic-review", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare:
        if args.revision_authorized or args.resume:
            parser.error("Preparation does not authorize training or resume")
        print(canonical(prepare(args.parent_run, args.output)))
        return 0
    if not args.revision_authorized:
        parser.error("Explicit approval of the revised protocol is required; previous budget approval alone is not that approval")
    return run(args.parent_run, args.output, revision_authorized=True, device=args.device,
        resume=args.resume, stop_after_ppo_steps=args.stop_after_ppo_steps,
        resume_after_diagnostic_review=args.resume_after_diagnostic_review)


if __name__ == "__main__":
    raise SystemExit(main())
