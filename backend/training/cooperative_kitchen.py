"""Kitchen MAPPO with exact resume and explicit on-policy learner masks.

Only actions sampled from the current Actor contribute to PPO's policy loss.
Scripted partner actions are never misrepresented as on-policy samples.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from env.cooperative_kitchen import CooperativeKitchen, KitchenConfig, OBSERVATION_FEATURES, program_decision
from backend.cooperative_kitchen.policy import ACTIONS, CHECKPOINT_SCHEMA, NumpyKitchenPolicy, export_checkpoint, forward, probabilities
from backend.cooperative_kitchen.torch_policy import SharedActor, CentralCritic, numpy_layers
from backend.cooperative_kitchen.splits import seeds


@dataclass
class TrainingConfig:
    seed: int = 0
    total_steps: int = 2_000_000
    n_envs: int = 16
    rollout_steps: int = 128
    learning_rate: float = 3e-4
    gamma: float = .99
    gae_lambda: float = .95
    clip: float = .2
    epochs: int = 4
    minibatch_size: int = 1024
    entropy_coefficient: float = .02
    value_coefficient: float = .5
    max_grad_norm: float = .5
    checkpoint_interval: int = 50_000
    validation_episodes: int = 24
    train_scenarios: int = 512
    selfplay_fraction: float = .5
    device: str = "mps"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def environment_signature():
    root = Path(__file__).resolve().parents[2] / "env" / "cooperative_kitchen"
    return hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in sorted(root.glob("*.py")))).hexdigest()


def training_counter_summary(log_path):
    """Recover run totals from explicitly scoped or legacy resume-local logs.

    This is reporting only: neither optimization nor rollout state reads these
    counters. Legacy boundaries are identified by a counter/step decrease.
    """
    path = Path(log_path)
    segments = []
    previous = None
    for line in path.read_text().splitlines() if path.exists() else ():
        if not line.strip():
            continue
        row = json.loads(line)
        marker = row.get("segment_start_joint_steps")
        boundary = previous is None or (
            marker is not None and marker != previous.get("segment_start_joint_steps")
        ) or (previous is not None and (
            row["joint_steps"] <= previous["joint_steps"]
            or row["episodes"] < previous["episodes"]
            or row["successes"] < previous["successes"]
        ))
        if boundary:
            segments.append({"start_joint_steps": marker if marker is not None else (previous["joint_steps"] if previous else 0),
                             "boundary_source": "explicit_segment_marker" if marker is not None else "legacy_counter_reset_inference",
                             "records": 0})
        segments[-1].update(end_joint_steps=row["joint_steps"], completed_episodes=row["episodes"],
                            successful_episodes=row["successes"], records=segments[-1]["records"] + 1)
        previous = row
    return {"schema": "cooperative_kitchen_training_counter_summary_v1", "source": str(path),
            "raw_counter_scope": "resume_segment", "segments": segments,
            "completed_episodes_across_segments": sum(row["completed_episodes"] for row in segments),
            "successful_episodes_across_segments": sum(row["successful_episodes"] for row in segments),
            "interpretation": "Sum segment-end counters; each original episodes/successes field counts only its training invocation"}


def gae(reward, value, done, bootstrap, gamma=.99, lam=.95):
    advantage = np.zeros_like(reward)
    last = np.zeros_like(bootstrap)
    for t in reversed(range(len(reward))):
        alive = 1 - done[t, :, None]
        following = bootstrap if t == len(reward) - 1 else value[t + 1]
        delta = reward[t] + gamma * following * alive - value[t]
        last = delta + gamma * lam * alive * last
        advantage[t] = last
    return advantage, advantage + value


def choose_partner(rng, selfplay_fraction):
    if rng.random() < selfplay_fraction:
        return {"actor": None, "profile": "selfplay"}
    return {"actor": rng.choice(("human", "ai")),
            "profile": rng.choices(("efficient", "upper", "lower", "perturbed"), weights=(4, 2, 2, 2))[0]}


def save_checkpoint(path, actor, critic, optimizer, config, steps, rng, reset_rng, envs, partners):
    payload = {"schema": CHECKPOINT_SCHEMA, "seed": config.seed, "joint_steps": steps,
               "config": asdict(config), "feature_names": list(OBSERVATION_FEATURES),
               "environment_signature": environment_signature(),
               "actor": {k: v.detach().cpu() for k, v in actor.state_dict().items()},
               "critic": {k: v.detach().cpu() for k, v in critic.state_dict().items()},
               "optimizer": optimizer.state_dict(), "numpy_rng": rng.bit_generator.state,
               "reset_rng": reset_rng.getstate(), "torch_rng": torch.get_rng_state(),
               "environments": [env.snapshot() for env in envs], "partners": partners,
               "training_kind": "MAPPO", "program_feedback": False,
               "training_scenario_seeds": list(seeds("train", config.train_scenarios, config.seed))}
    if config.device == "mps":
        payload["mps_rng"] = torch.mps.get_rng_state()
    temporary = Path(path).with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def validate_resume(payload, config):
    if payload.get("schema") != CHECKPOINT_SCHEMA or payload.get("seed") != config.seed:
        raise ValueError("Checkpoint schema or training seed mismatch")
    if payload.get("environment_signature") != environment_signature():
        raise ValueError("Kitchen environment changed; start a new training version")
    if tuple(payload.get("feature_names", [])) != tuple(OBSERVATION_FEATURES):
        raise ValueError("Kitchen observation contract changed")
    saved = payload["config"]
    for key, value in asdict(config).items():
        if key not in {"total_steps", "device", "validation_episodes"} and saved.get(key) != value:
            raise ValueError(f"Resume training parameter changed: {key}")
    if len(payload.get("environments", [])) != config.n_envs or len(payload.get("partners", [])) != config.n_envs:
        raise ValueError("Incomplete environment or partner snapshots")


def train(config, output, resume=None, validate=True):
    if config.total_steps % config.n_envs or config.checkpoint_interval % config.n_envs:
        raise ValueError("Budgets must be divisible by the number of environments")
    if config.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available; explicitly select --device cpu")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    reset_rng = random.Random(config.seed)
    pool = list(seeds("train", config.train_scenarios, config.seed))
    envs = [CooperativeKitchen(seed=pool[i % len(pool)], scenario_id="generated") for i in range(config.n_envs)]
    partners = [choose_partner(reset_rng, config.selfplay_fraction) for _ in envs]
    width, state_width = len(OBSERVATION_FEATURES), len(envs[0].global_state())
    actor, critic = SharedActor(width).to(config.device), CentralCritic(state_width).to(config.device)
    optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=config.learning_rate, eps=1e-5)
    steps = 0
    if resume:
        payload = torch.load(resume, map_location=config.device, weights_only=False)
        validate_resume(payload, config)
        actor.load_state_dict(payload["actor"]); critic.load_state_dict(payload["critic"])
        optimizer.load_state_dict(payload["optimizer"])
        rng.bit_generator.state = payload["numpy_rng"]
        reset_rng.setstate(payload["reset_rng"])
        torch.set_rng_state(payload["torch_rng"].cpu())
        if config.device == "mps" and "mps_rng" in payload:
            torch.mps.set_rng_state(payload["mps_rng"].cpu())
        for env, snapshot in zip(envs, payload["environments"]):
            env.restore(snapshot)
        partners = payload["partners"]
        steps = int(payload["joint_steps"])
    started, initial = time.perf_counter(), steps
    next_checkpoint = (steps // config.checkpoint_interval + 1) * config.checkpoint_interval
    episodes = wins = 0
    with (output / "training.jsonl").open("a") as log:
        while steps < config.total_steps:
            horizon = min(config.rollout_steps, (min(next_checkpoint, config.total_steps) - steps) // config.n_envs)
            shape = (horizon, config.n_envs, 2)
            ob = np.zeros((*shape, width), np.float32)
            st = np.zeros((horizon, config.n_envs, state_width), np.float32)
            act = np.zeros(shape, np.int64)
            old_log = np.zeros(shape, np.float32)
            reward = np.zeros(shape, np.float32)
            value = np.zeros(shape, np.float32)
            learner = np.ones(shape, np.float32)
            done = np.zeros((horizon, config.n_envs), np.float32)
            actor_layers, critic_layers = numpy_layers(actor), numpy_layers(critic)
            for t in range(horizon):
                observed = [env.observations() for env in envs]
                ob[t] = [[o["human"], o["ai"]] for o in observed]
                st[t] = [env.global_state(obs) for env, obs in zip(envs, observed)]
                logits = forward(ob[t], actor_layers)
                probs = probabilities(logits)
                choices = (rng.random((config.n_envs, 2, 1)) > probs.cumsum(axis=-1)).sum(axis=-1).clip(0, 5)
                act[t] = choices
                old_log[t] = np.log(np.take_along_axis(probs, choices[..., None], axis=-1)[..., 0].clip(1e-12))
                value[t] = forward(st[t], critic_layers)
                for i, env in enumerate(envs):
                    actions = {key: ACTIONS[choices[i, j]] for j, key in enumerate(("human", "ai"))}
                    partner = partners[i]
                    if partner["actor"]:
                        key = partner["actor"]
                        actions[key] = program_decision(env, key, profile=partner["profile"], rng=reset_rng)["action"]
                        learner[t, i, 0 if key == "human" else 1] = 0
                    result = env.step(actions, include_state=False)
                    reward[t, i] = [result["rewards"][key] for key in ("human", "ai")]
                    done[t, i] = bool(result["done"])
                    if result["done"]:
                        episodes += 1
                        wins += env.public_view()["orders"] >= 2
                        env.reset(seed=reset_rng.choice(pool), scenario_id="generated")
                        partners[i] = choose_partner(reset_rng, config.selfplay_fraction)
                steps += config.n_envs
            bootstrap = forward(np.asarray([env.global_state() for env in envs], np.float32), critic_layers)
            advantages, returns = gae(reward, value, done, bootstrap, config.gamma, config.gae_lambda)
            mask = learner.reshape(-1)
            flat_adv = advantages.reshape(-1)
            valid_adv = flat_adv[mask > .5]
            flat_adv = (flat_adv - valid_adv.mean()) / (valid_adv.std() + 1e-8)
            device = config.device
            obs_t = torch.as_tensor(ob.reshape(-1, width), device=device)
            states_t = torch.as_tensor(np.repeat(st[:, :, None, :], 2, axis=2).reshape(-1, state_width), device=device)
            actions_t = torch.as_tensor(act.reshape(-1), device=device)
            old_log_t = torch.as_tensor(old_log.reshape(-1), device=device)
            adv_t = torch.as_tensor(flat_adv, device=device)
            ret_t = torch.as_tensor(returns.reshape(-1), device=device)
            old_values_t = torch.as_tensor(value.reshape(-1), device=device)
            mask_t = torch.as_tensor(mask, device=device)
            identities = torch.arange(len(mask), device=device) % 2
            losses, entropies = [], []
            for _ in range(config.epochs):
                chunks = max(1, (len(mask) + config.minibatch_size - 1) // config.minibatch_size)
                for indices in np.array_split(rng.permutation(len(mask)), chunks):
                    idx = torch.as_tensor(indices, device=device)
                    distribution = torch.distributions.Categorical(logits=actor(obs_t[idx]))
                    ratio = (distribution.log_prob(actions_t[idx]) - old_log_t[idx]).exp()
                    weights = mask_t[idx]; denominator = weights.sum().clamp(min=1)
                    policy_loss = -(torch.minimum(ratio * adv_t[idx], ratio.clamp(1 - config.clip, 1 + config.clip) * adv_t[idx]) * weights).sum() / denominator
                    predicted = critic(states_t[idx]).gather(1, identities[idx, None])[:, 0]
                    clipped = old_values_t[idx] + (predicted - old_values_t[idx]).clamp(-config.clip, config.clip)
                    value_loss = .5 * torch.maximum((predicted - ret_t[idx]) ** 2, (clipped - ret_t[idx]) ** 2).mean()
                    entropy = (distribution.entropy() * weights).sum() / denominator
                    loss = policy_loss + config.value_coefficient * value_loss - config.entropy_coefficient * entropy
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    nn.utils.clip_grad_norm_([*actor.parameters(), *critic.parameters()], config.max_grad_norm)
                    optimizer.step(); losses.append(loss.detach()); entropies.append(entropy.detach())
            record = {"joint_steps": steps, "episodes": episodes, "successes": int(wins),
                      "counter_scope": "resume_segment", "segment_start_joint_steps": initial,
                      "loss": float(torch.stack(losses).mean().cpu()), "entropy": float(torch.stack(entropies).mean().cpu()),
                      "steps_per_second": (steps - initial) / max(.001, time.perf_counter() - started),
                      "on_policy_rows": int(mask.sum()), "partner_rows": int((1 - mask).sum())}
            log.write(json.dumps(record) + "\n"); log.flush()
            if steps >= next_checkpoint or steps >= config.total_steps:
                path = output / f"checkpoint_{steps:09d}.pt"
                save_checkpoint(path, actor, critic, optimizer, config, steps, rng, reset_rng, envs, partners)
                policy = export_checkpoint(path, output / f"actor_{steps:09d}.npz")
                metrics = {}
                if validate:
                    from .cooperative_kitchen_validation import evaluate_policy
                    metrics = evaluate_policy(policy, split="validation", episodes=config.validation_episodes)
                    atomic_json(output / f"validation_{steps:09d}.json", metrics)
                manifest = {"schema": "cooperative_kitchen_training_candidate_v1", "status": "candidate", "seed": config.seed,
                            "joint_steps": steps, "checkpoint": path.name, "actor": policy.path.name,
                            "checkpoint_sha256": policy.checkpoint_id, "actor_sha256": policy.artifact_sha256,
                            "environment_signature": environment_signature(), "config": asdict(config),
                            "training_gate": bool(metrics.get("training_gate")), "validation": metrics,
                            "training_counter_summary": training_counter_summary(output / "training.jsonl"),
                            "rcpd_feedback": False, "sampling_device": "cpu", "optimizer_device": config.device}
                atomic_json(output / "manifest.json", manifest)
                print(json.dumps({**record, "checkpoint": str(path), "validation": metrics.get("summary", {})}), flush=True)
                next_checkpoint += config.checkpoint_interval
    return output / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--checkpoint-interval", type=int, default=50_000)
    parser.add_argument("--validation-episodes", type=int, default=24)
    parser.add_argument("--output", default="output/cooperative_kitchen/v1/seed_0")
    parser.add_argument("--resume")
    parser.add_argument("--no-validation", action="store_true")
    args = parser.parse_args()
    cfg = TrainingConfig(seed=args.seed, total_steps=args.steps, n_envs=args.envs, rollout_steps=args.rollout,
                         device=args.device, checkpoint_interval=args.checkpoint_interval, validation_episodes=args.validation_episodes)
    train(cfg, args.output, args.resume, not args.no_validation)


if __name__ == "__main__":
    main()
