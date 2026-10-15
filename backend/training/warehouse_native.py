"""Native MAPPO: five unmasked actions, shared Actor and centralized Critic.

The only authorized long run is the 500k pure-RL foundation. Feedback/control
continuations have explicit modes and require --additional-budget-authorized.
Program partner actions never enter the Actor objective. Every sampled joint
transition, including throughput measurement, counts toward the run budget.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import random
import signal
import shutil
import tempfile
import time

import numpy as np
import torch

from env.warehouse.navigation import ACTIONS
from env.warehouse_native.environment import NativeWarehouseEnv
from env.warehouse_native.policy import NativeActorCritic, NumPyNativeActor
from env.warehouse_native.partners import partner_action
from env.warehouse_native.scenarios import generate_manifest, reset_scenario
from .warehouse_native_common import (
    ROOT, VERSION, atomic_json, append_jsonl, canonical, digest, file_hash,
    jsonable, load_protocol, preserve_inventory, source_hashes,
)
from .warehouse_native_evaluation import evaluate, capability


def gae(rewards, values, dones, bootstrap, gamma=.99, lam=.95):
    """Time-major [T, environment, role]; terminal values never cross resets."""
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last = np.zeros_like(bootstrap, dtype=np.float32)
    for t in reversed(range(len(rewards))):
        following = bootstrap if t + 1 == len(rewards) else values[t + 1]
        alive = 1. - dones[t, :, None]
        delta = rewards[t] + gamma * following * alive - values[t]
        last = delta + gamma * lam * alive * last
        advantages[t] = last
    return advantages, advantages + values


def ppo_actor_objective(logits, actions, old_log_probs, advantages, trainable, *, clip=.2, entropy=.01):
    """No loss/entropy/gradient from a program-controlled or inactive role."""
    selected = trainable.bool()
    if not selected.any():
        return logits.sum() * 0., {"actor_loss": 0., "entropy": 0., "trainable_rows": 0}
    dist = torch.distributions.Categorical(logits=logits[selected])
    log_probs = dist.log_prob(actions[selected])
    ratio = (log_probs - old_log_probs[selected]).exp()
    adv = advantages[selected]
    policy_loss = -torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
    ent = dist.entropy().mean()
    return policy_loss - entropy * ent, {"actor_loss": float(policy_loss.detach()),
        "entropy": float(ent.detach()), "trainable_rows": int(selected.sum()),
        "approx_kl": float(((ratio - 1) - (log_probs - old_log_probs[selected])).mean().detach())}


def atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temp)
        with open(temp, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def acknowledge_update(output, payload):
    """Recover a checkpoint-first commit interrupted before its journal flush.

    Each latest checkpoint carries the one completed update awaiting journal
    acknowledgement. Duplicate appends are rejected by transition counters.
    """
    pending = payload.get("acknowledgement")
    if not pending:
        return
    log = Path(output) / "training.jsonl"
    logged = {json.loads(line)["joint_steps"] for line in log.read_text().splitlines() if line.strip()} if log.exists() else set()
    if pending["record"]["joint_steps"] not in logged:
        append_jsonl(log, pending["record"])
    episodes = Path(output) / "episodes.jsonl"
    recorded = {json.loads(line)["joint_step"] for line in episodes.read_text().splitlines() if line.strip()} if episodes.exists() else set()
    for row in pending["episodes"]:
        if row["joint_step"] not in recorded:
            append_jsonl(episodes, row)
            recorded.add(row["joint_step"])


def reserve_sampling(path, amount, *, cap):
    """Durable upper bound on attempted sampling, including a crashed rollout.

    After a hard crash an unfinished reservation is conservatively spent. Thus
    replaying the last complete optimizer state cannot exceed the user budget.
    """
    path = Path(path)
    record = json.loads(path.read_text()) if path.exists() else {"cap": cap, "reserved_joint_steps": 0}
    if record["cap"] != cap or amount < 0 or record["reserved_joint_steps"] + amount > cap:
        raise ValueError("Sampling budget would be exceeded")
    record["reserved_joint_steps"] += amount
    atomic_json(path, record)
    return record["reserved_joint_steps"]


class NativeTrainer:
    def __init__(self, protocol, scenarios, *, device="cpu", feedback=None):
        self.protocol, self.scenarios = copy.deepcopy(protocol), copy.deepcopy(scenarios)
        self.cfg = protocol["training"]
        self.device = torch.device(device)
        self.seed = protocol["seed"]
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.rng = np.random.default_rng(self.seed)
        self.envs = [NativeWarehouseEnv() for _ in range(self.cfg["environments"])]
        self.partner_kinds = [""] * len(self.envs)
        self.program_roles = [-1] * len(self.envs)
        self.scenario_ids = [""] * len(self.envs)
        self.episode_returns = np.zeros(len(self.envs), dtype=np.float64)
        self.completed_episodes = []
        self.episode_count = 0
        self.joint_steps = 0
        self.optimizer_updates = 0
        self.minibatch_updates = 0
        self.best = None
        self.elapsed_seconds = 0.
        self.feedback = feedback
        self.feedback_enabled = feedback is not None
        self.feedback_replay = []
        for i in range(len(self.envs)):
            self.reset_one(i)
        sample = self.envs[0]
        self.model = NativeActorCritic(len(sample.observations()["robot_1"]), len(sample.global_state())).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg["learning_rate"], eps=1e-5)

    def reset_one(self, i):
        choices = self.scenarios["splits"]["train"]
        entry = choices[int(self.rng.integers(len(choices)))]
        reset_scenario(self.envs[i], entry)
        self.scenario_ids[i] = entry["id"]
        mix = self.protocol["partners"]
        self.partner_kinds[i] = str(self.rng.choice(list(mix), p=list(mix.values())))
        self.program_roles[i] = -1 if self.partner_kinds[i] == "selfplay" else int(self.rng.integers(2))
        self.episode_returns[i] = 0.
        self.episode_count += 1

    def arrays(self):
        obs = np.stack([np.stack(list(e.observations().values())) for e in self.envs])
        state = np.stack([e.global_state() for e in self.envs])
        return obs, state

    def infer(self, obs, state):
        n = len(obs)
        with torch.no_grad():
            logits = self.model.actor_logits(torch.as_tensor(obs.reshape(-1, obs.shape[-1]), device=self.device))
            probabilities = torch.softmax(logits, -1).cpu().numpy().reshape(n, 2, 5)
            expanded = np.repeat(state, 2, axis=0)
            roles = torch.as_tensor(np.tile([0, 1], n), device=self.device)
            values = self.model.values(torch.as_tensor(expanded, device=self.device), roles).cpu().numpy().reshape(n, 2)
        return probabilities, values

    def collect(self, time_steps):
        keys = ("observations", "states", "actions", "old_log_probs", "values", "rewards", "dones", "trainable")
        rows = {key: [] for key in keys}
        action_audit = {"neural_submitted": 0, "program_submitted": 0, "neural_overrides": 0}
        for _ in range(time_steps):
            obs, states = self.arrays()
            # Program decisions use S_t only and are made without access to logits/actions.
            partner_decisions = [None if role < 0 else partner_action(env, f"robot_{role+1}", kind, self.rng)
                                 for env, kind, role in zip(self.envs, self.partner_kinds, self.program_roles)]
            probabilities, values = self.infer(obs, states)
            draws = self.rng.random(probabilities.shape[:2])
            actions = (draws[..., None] > probabilities.cumsum(-1)).sum(-1).clip(0, 4)
            old_log_probs = np.log(np.take_along_axis(probabilities, actions[..., None], -1)[..., 0].clip(1e-12))
            trainable = np.ones(actions.shape, dtype=np.float32)
            rewards = np.zeros(actions.shape, dtype=np.float32)
            dones = np.zeros(len(self.envs), dtype=np.float32)
            for i, (env, role) in enumerate(zip(self.envs, self.program_roles)):
                for j, agent in enumerate(env.state.agents):
                    if not agent.active:
                        trainable[i, j] = 0
                submitted = {f"robot_{j+1}": ACTIONS[int(actions[i, j])] for j in range(2)}
                if role >= 0:
                    trainable[i, role] = 0
                    submitted[f"robot_{role+1}"] = partner_decisions[i]
                    action_audit["program_submitted"] += 1
                for j in range(2):
                    if trainable[i, j]:
                        assert submitted[f"robot_{j+1}"] == ACTIONS[int(actions[i, j])]
                        action_audit["neural_submitted"] += 1
                _, reward, terminated, truncated, _ = env.step(submitted)
                rewards[i] = [reward[f"robot_{j+1}"] for j in range(2)]
                self.episode_returns[i] += float(rewards[i].mean())
                self.joint_steps += 1
                if terminated or truncated:
                    dones[i] = 1
                    self.completed_episodes.append({"joint_step": self.joint_steps,
                        "scenario_id": self.scenario_ids[i], "partner": self.partner_kinds[i],
                        "program_role": role, "return": self.episode_returns[i],
                        "team_deliveries": env.state.total_deliveries,
                        "individual_deliveries": [a.deliveries_completed for a in env.state.agents],
                        "shutdowns": env.state.shutdown_count, "collisions": env.state.robot_collision_events,
                        "length": env.state.frame})
                    self.reset_one(i)
            for key, value in zip(keys, (obs, states, actions, old_log_probs, values, rewards, dones, trainable)):
                rows[key].append(value)
        rows = {key: np.asarray(value) for key, value in rows.items()}
        obs, state = self.arrays()
        _, bootstrap = self.infer(obs, state)
        rows["advantages"], rows["returns"] = gae(rows["rewards"], rows["values"], rows["dones"],
                                                  bootstrap, self.cfg["gamma"], self.cfg["gae_lambda"])
        rows["audit"] = action_audit
        return rows

    def update(self, batch):
        obs = batch["observations"].reshape(-1, self.model.obs_dim)
        states = np.repeat(batch["states"][:, :, None, :], 2, axis=2).reshape(-1, self.model.state_dim)
        roles = np.tile([0, 1], len(obs) // 2)
        trainable = batch["trainable"].reshape(-1)
        adv = batch["advantages"].reshape(-1).copy()
        selected = trainable > 0
        if selected.any():
            adv[selected] = (adv[selected] - adv[selected].mean()) / (adv[selected].std() + 1e-8)
        data = {"obs": obs, "states": states, "roles": roles, "trainable": trainable,
                "actions": batch["actions"].reshape(-1), "old": batch["old_log_probs"].reshape(-1),
                "adv": adv, "returns": batch["returns"].reshape(-1)}
        data = {key: torch.as_tensor(value, device=self.device) for key, value in data.items()}
        metrics = []
        for _ in range(self.cfg["epochs"]):
            permutation = self.rng.permutation(len(obs))
            for start in range(0, len(obs), self.cfg["minibatch"]):
                ix = torch.as_tensor(permutation[start:start + self.cfg["minibatch"]], device=self.device)
                logits = self.model.actor_logits(data["obs"][ix])
                actor_loss, m = ppo_actor_objective(logits, data["actions"][ix], data["old"][ix],
                    data["adv"][ix], data["trainable"][ix], clip=self.cfg["clip"], entropy=self.cfg["entropy"])
                values = self.model.values(data["states"][ix], data["roles"][ix])
                critic_loss = .5 * (values - data["returns"][ix]).square().mean()
                feedback_loss = logits.sum() * 0.
                feedback_metrics = {"feedback_loss": 0., "feedback_gradient_norm": 0.}
                if self.feedback_enabled and data["trainable"][ix].bool().any():
                    sel = data["trainable"][ix].bool()
                    feedback_loss, feedback_metrics = self.feedback.loss(logits[sel], data["obs"][ix][sel].detach().cpu().numpy())
                    if feedback_loss.requires_grad:
                        grads = torch.autograd.grad(feedback_loss, tuple(self.model.actor.parameters()),
                                                    retain_graph=True, allow_unused=True)
                        feedback_metrics["feedback_gradient_norm"] = float(torch.sqrt(sum((g.square().sum() for g in grads if g is not None), logits.new_zeros(()))).detach())
                self.optimizer.zero_grad(set_to_none=True)
                loss = actor_loss + critic_loss + feedback_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite native PPO loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["gradient_norm"])
                self.optimizer.step()
                self.minibatch_updates += 1
                metrics.append({**m, "critic_loss": float(critic_loss.detach()), "gradient_norm": float(norm),
                                **{k: v for k, v in feedback_metrics.items() if isinstance(v, (int, float))}})
        self.optimizer_updates += 1
        return {key: float(np.mean([m.get(key, 0.) for m in metrics])) for key in set().union(*(m.keys() for m in metrics))}

    def state_dict(self):
        result = {"version": VERSION, "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "obs_dim": self.model.obs_dim, "state_dim": self.model.state_dim, "protocol": self.protocol,
            "scenario_manifest_hash": digest(self.scenarios), "sources": source_hashes(),
            "joint_steps": self.joint_steps, "optimizer_updates": self.optimizer_updates,
            "minibatch_updates": self.minibatch_updates, "rng": copy.deepcopy(self.rng.bit_generator.state),
            "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
            "envs": [e.snapshot() for e in self.envs], "partner_kinds": self.partner_kinds,
            "program_roles": self.program_roles, "scenario_ids": self.scenario_ids,
            "episode_returns": self.episode_returns, "episode_count": self.episode_count,
            "best": self.best, "elapsed_seconds": self.elapsed_seconds,
            "feedback_enabled": self.feedback_enabled,
            "feedback_state": self.feedback.state_dict() if self.feedback else None}
        if self.device.type == "mps":
            result["mps_rng"] = torch.mps.get_rng_state()
        return result

    def load_state_dict(self, payload, *, verify_sources=True):
        if payload["version"] != VERSION or payload["protocol"] != self.protocol:
            raise ValueError("Checkpoint version/protocol mismatch")
        if payload["scenario_manifest_hash"] != digest(self.scenarios):
            raise ValueError("Checkpoint scene manifest mismatch")
        if verify_sources and payload["sources"] != source_hashes():
            raise ValueError("Training code changed; start an explicitly versioned continuation")
        if bool(payload["feedback_enabled"]) != self.feedback_enabled:
            raise ValueError("Feedback mode mismatch requires an explicit fork")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        for key in ("joint_steps", "optimizer_updates", "minibatch_updates", "partner_kinds", "program_roles",
                    "scenario_ids", "episode_returns", "episode_count", "best", "elapsed_seconds"):
            setattr(self, key, copy.deepcopy(payload[key]))
        for env, saved in zip(self.envs, payload["envs"]):
            env.restore(saved)
        self.rng.bit_generator.state = copy.deepcopy(payload["rng"])
        random.setstate(payload["python_rng"])
        np.random.set_state(payload["numpy_rng"])
        torch.set_rng_state(payload["torch_rng"].cpu())
        if self.device.type == "mps" and "mps_rng" in payload:
            torch.mps.set_rng_state(payload["mps_rng"].cpu())
        if self.feedback:
            self.feedback.load_state_dict(payload["feedback_state"])

    def export(self, path):
        return self.model.export_npz(path, {"experiment_version": VERSION, "joint_steps": self.joint_steps,
            "optimizer_updates": self.optimizer_updates, "protocol_sha256": digest(self.protocol),
            "scenario_manifest_sha256": digest(self.scenarios), "source_sha256": digest(source_hashes()),
            "initialization": "fresh_random", "candidate": True,
            "feedback_enabled": self.feedback_enabled, "feature_names": self.envs[0].feature_names})


def select_device(requested):
    if requested != "auto":
        return requested, {"requested": requested}
    # Kernel benchmark is synthetic and performs no environment transitions.
    durations = {}
    for device in (["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"]):
        torch.manual_seed(0)
        net = NativeActorCritic(200, 400).to(device)
        obs = torch.zeros(32, 200, device=device)
        start = time.perf_counter()
        for i in range(30):
            with torch.no_grad():
                net.actor_logits(obs).cpu().numpy()
        durations[device] = time.perf_counter() - start
    chosen = min(durations, key=durations.get)
    return chosen, {"synthetic_inference_seconds": durations, "chosen": chosen,
                    "actual_end_to_end_throughput_measured_in_counted_rollouts": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--steps", type=int, default=500000)
    parser.add_argument("--stop-after", type=int, help="Stop/save at this global step, for an intentional resumable measurement")
    parser.add_argument("--mode", choices=["foundation", "feedback", "control"], default="foundation")
    parser.add_argument("--additional-budget-authorized", action="store_true")
    args = parser.parse_args(argv)
    protocol = load_protocol()
    cap = protocol["authorized_run"]["joint_steps_max"]
    if args.mode != "foundation" or args.steps > cap:
        if not args.additional_budget_authorized:
            parser.error("Only one 500k foundation run is authorized; feedback/control need an explicit additional budget")
        # The orchestration for a paired continuation is added separately; never
        # silently run a foundation job while labeling it feedback.
        parser.error("Use the versioned paired-continuation entry after the foundation capability report")
    n = protocol["training"]["environments"]
    if args.steps < 1 or args.steps % n or (args.stop_after is not None and (args.stop_after % n or args.stop_after > args.steps)):
        parser.error("Budgets must be positive multiples of the number of environments")
    torch.set_num_threads(1)
    device, benchmark = select_device(args.device)
    output = args.output.resolve()
    if args.resume:
        manifest = json.loads((output / "scenarios.json").read_text())
        if not (output / "run.json").exists():
            parser.error("Resume requires the original run directory")
        previous_run = json.loads((output / "run.json").read_text())
        if previous_run["budget"] != args.steps:
            parser.error("Resume must preserve the original authorized budget")
    else:
        if output.exists() and any(output.iterdir()):
            parser.error("Fresh run requires an empty output directory; resume instead of overwriting")
        output.mkdir(parents=True, exist_ok=True)
    # Prevent two resumed processes from consuming the same sampler/budget.
    run_lock = (output / "run.lock").open("a+")
    try:
        fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("This training run is already live; do not start a duplicate")
    if not args.resume:
        preserve_inventory(output / "original_artifact_hashes.json")
        manifest = generate_manifest(counts=protocol["splits"], seed=protocol["seed"])
        atomic_json(output / "scenarios.json", manifest)
        atomic_json(output / "protocol.json", protocol)
        for relative in source_hashes():
            destination = output / "source_bundle" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        atomic_json(output / "run.json", {"version": VERSION, "status": "foundation_running", "device": device,
            "device_probe": benchmark, "budget": args.steps, "feedback_enabled": False,
            "sources": source_hashes(), "protocol_sha256": digest(protocol),
            "scene_sha256": digest(manifest), "pid": os.getpid()})
        reserve_sampling(output / "sampling_budget.json", 0, cap=args.steps)
    trainer = NativeTrainer(protocol, manifest, device=device)
    if args.resume:
        payload = torch.load(args.resume, map_location=device, weights_only=False)
        log = output / "training.jsonl"
        if log.exists():
            logged_steps = [json.loads(line)["joint_steps"] for line in log.read_text().splitlines() if line.strip()]
            if logged_steps and max(logged_steps) > payload["joint_steps"]:
                parser.error("Newer acknowledged training updates exist; resume their checkpoint, do not replay an older one")
        trainer.load_state_dict(payload)
        acknowledge_update(output, payload)
    else:
        trainer.export(output / "initial_actor.npz")
        atomic_torch_save(output / "initial_checkpoint.pt", trainer.state_dict())
    requested_target = args.stop_after if args.stop_after is not None else args.steps
    budget_record = json.loads((output / "sampling_budget.json").read_text())
    reserved = budget_record["reserved_joint_steps"]
    if reserved < trainer.joint_steps:
        parser.error("Sampling ledger is older than checkpoint; budget provenance is inconsistent")
    discarded_upper_bound = reserved-trainer.joint_steps
    target = min(requested_target, trainer.joint_steps + args.steps-reserved)
    if trainer.joint_steps >= target:
        parser.error("Checkpoint already reached the requested stop; refusing to repeat transitions")
    for actor_kind in ("reference", "random"):
        path = output / f"validation_{actor_kind}.json"
        if not path.exists():
            start = time.perf_counter()
            result = evaluate(actor_kind, manifest["splits"]["validation"])
            atomic_json(path, result)
            print(canonical({"event": "baseline_complete", "kind": actor_kind, "seconds": time.perf_counter()-start,
                             "summary": result["summary"]}), flush=True)
    reference = json.loads((output / "validation_reference.json").read_text())
    random_report = json.loads((output / "validation_random.json").read_text())
    checkpoint_every = protocol["training"]["checkpoint_interval"]
    started = time.perf_counter()
    starting_step = trainer.joint_steps
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    old_sigint = signal.signal(signal.SIGINT, request_stop)
    old_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        while trainer.joint_steps < target:
            next_checkpoint = ((trainer.joint_steps // checkpoint_every) + 1) * checkpoint_every
            boundary = min(target, next_checkpoint)
            length = min(protocol["training"]["rollout_steps"], (boundary-trainer.joint_steps)//n)
            tick = time.perf_counter()
            reserved = reserve_sampling(output / "sampling_budget.json", length*n, cap=args.steps)
            batch = trainer.collect(length)
            collected = time.perf_counter()
            metrics = trainer.update(batch)
            elapsed = time.perf_counter() - tick
            trainer.elapsed_seconds += elapsed
            record = {"joint_steps": trainer.joint_steps, "optimizer_updates": trainer.optimizer_updates,
                "collection_seconds": collected-tick, "update_seconds": elapsed-(collected-tick),
                "joint_steps_per_second": length*n/elapsed, "metrics": metrics, "execution_audit": batch["audit"],
                "reserved_joint_steps_upper_bound": reserved, "discarded_after_crash_upper_bound": discarded_upper_bound}
            # Save all environment/optimizer/RNG state before confirming this
            # update in the journal. This survives a real process crash, too.
            checkpoint_payload = trainer.state_dict()
            checkpoint_payload["acknowledgement"] = {"record": record, "episodes": list(trainer.completed_episodes)}
            atomic_torch_save(output / "latest_checkpoint.pt", checkpoint_payload)
            acknowledge_update(output, checkpoint_payload)
            trainer.completed_episodes.clear()
            if trainer.optimizer_updates == 2:
                rate = (trainer.joint_steps-starting_step)/(time.perf_counter()-started)
                print(canonical({"event": "counted_throughput", "joint_steps": trainer.joint_steps,
                    "joint_steps_per_second": rate, "remaining_training_seconds_estimate": (args.steps-trainer.joint_steps)/rate,
                    "validation_time_additional": True}), flush=True)
            if trainer.joint_steps == boundary:
                actor_path = output / f"checkpoints/actor_{trainer.joint_steps:07d}.npz"
                trainer.export(actor_path)
                actor = NumPyNativeActor(actor_path)
                validation_start = time.perf_counter()
                report = evaluate(actor, manifest["splits"]["validation"])
                report["capability"] = capability(report, reference, random_report, protocol)
                report["joint_steps"] = trainer.joint_steps
                report["actor_sha256"] = actor.artifact_sha256
                report["seconds"] = time.perf_counter()-validation_start
                counted_wall = time.perf_counter()-started
                report["throughput"] = {"joint_steps_counted_this_process": trainer.joint_steps-starting_step,
                    "seconds_including_sampling_updates_persistence_and_nn_validation": counted_wall,
                    "joint_steps_per_second_including_nn_validation": (trainer.joint_steps-starting_step)/counted_wall,
                    "estimated_remaining_training_seconds": (args.steps-reserved) * trainer.elapsed_seconds / max(1, trainer.joint_steps),
                    "estimated_remaining_validation_seconds": ((args.steps-reserved + checkpoint_every-1)//checkpoint_every)*report["seconds"]}
                score = float(np.mean([s["mean_team_deliveries"] for s in report["summary"].values()]))
                shutdowns = float(np.mean([s["mean_shutdowns"] for s in report["summary"].values()]))
                candidate = {"score": score, "shutdowns": shutdowns, "joint_steps": trainer.joint_steps,
                             "actor": str(actor_path.relative_to(output)), "actor_sha256": actor.artifact_sha256,
                             "capability": report["capability"]}
                if trainer.best is None or (score, -shutdowns, -trainer.joint_steps) > (
                        trainer.best["score"], -trainer.best["shutdowns"], -trainer.best["joint_steps"]):
                    trainer.best = candidate
                atomic_json(output / f"validation/step_{trainer.joint_steps:07d}.json", report)
                atomic_torch_save(output / f"checkpoints/checkpoint_{trainer.joint_steps:07d}.pt", trainer.state_dict())
                atomic_torch_save(output / "latest_checkpoint.pt", trainer.state_dict())
                atomic_json(output / "progress.json", {"status": "running" if trainer.joint_steps < target else "paused" if target<args.steps else "foundation_budget_complete",
                    "joint_steps": trainer.joint_steps, "authorized_budget": args.steps, "best": trainer.best,
                    "elapsed_training_seconds": trainer.elapsed_seconds, "latest": candidate,
                    "reserved_joint_steps_upper_bound": reserved,
                    "discarded_after_crash_upper_bound": discarded_upper_bound})
                print(canonical({"event": "checkpoint", "joint_steps": trainer.joint_steps, "best": trainer.best,
                                 "validation_seconds": report["seconds"], "throughput": report["throughput"]}), flush=True)
            if stop_requested:
                atomic_torch_save(output / "interrupted_checkpoint.pt", trainer.state_dict())
                print(canonical({"event": "interrupted_saved_after_complete_update", "joint_steps": trainer.joint_steps}), flush=True)
                return 130
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        signal.signal(signal.SIGTERM, old_sigterm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
