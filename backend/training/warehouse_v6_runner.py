"""One parameterized, paired Warehouse v6 BC + constrained PPO runner.

This module trains a raw and a regularized dual Actor from the same offline
BC tensors and the same 120-step scenario schedule. During self-play, both
actions are sampled from their own NN logits. The offline BC teacher and
programs never enter the environment action path.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from hashlib import sha256
import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from env.warehouse.domain import participant_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import AGENTS, action_masks
from env.warehouse.public_observation_v6 import (
    FEATURE_SCHEMA_SHA256, OBSERVATION_DIM, OBSERVATION_VERSION,
    encode_public_observation,
)
from .warehouse_v6_actor import (
    MODEL_VERSION, FrozenWarehouseV6Actor, WarehouseV6Actor, export_actor,
    policy_features_torch, role_hashes,
)
from .warehouse_v6_program import (
    WarehouseV6Program, fit_program, audit_frozen_program,
)
from .warehouse_v6_teacher import public_bc_teacher


RUNNER_VERSION = "warehouse-v6-paired-cost-ppo.v2"
SCENARIO_BLOCK_STEPS = 120
PROGRAM_LAMBDA = 0.01
PROGRAM_START_FRACTION = 0.80
PROGRAM_RAMP_FRACTION = 0.05
PROGRAM_BUFFER_STATES = 20_000
PROGRAM_SPLIT_EPISODES = (50, 25, 25, 24)
PROGRAM_MIN_EPISODES = sum(PROGRAM_SPLIT_EPISODES)
TRAINING_REWARD_OVERRIDES = {"robot_collision_points": -100.0,
                             "shutdown_points": -200.0}
DEFAULT_BC_EPISODES = 200
DEFAULT_BC_EPOCHS = 30
DEFAULT_DAGGER_ROUNDS = 4
DEFAULT_DAGGER_EPISODES = 80
DEFAULT_DAGGER_EPOCHS = 10

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_FILES = (
    *(path.relative_to(_SOURCE_ROOT).as_posix()
      for path in sorted((_SOURCE_ROOT / "env" / "warehouse").glob("*.py"))),
    "core/policy_contracts.py",
    "backend/training/warehouse_v6_actor.py",
    "backend/training/warehouse_v6_program.py",
    "backend/training/warehouse_v6_runner.py",
    "backend/training/warehouse_v6_teacher.py",
    "experiments/warehouse/protocol.py",
    "configs/study_v3_warehouse_neural_v6.json",
)


def source_closure_files_sha256() -> dict[str, str]:
    """Hash the complete Warehouse training, physics, and protocol closure."""
    return {relative: sha256((_SOURCE_ROOT / relative).read_bytes()).hexdigest()
            for relative in sorted(set(_SOURCE_FILES))}


def source_closure_sha256() -> str:
    """Stable code identity shared by the run spec, checkpoints, and admission."""
    return _digest_json(source_closure_files_sha256())


def _assert_source_closure(expected: str) -> None:
    if source_closure_sha256() != expected:
        raise RuntimeError("Warehouse v6 source closure changed during training")


def _digest_json(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def _checked_spec(run_spec: Mapping) -> dict:
    required = {"candidate_id", "pair_seed", "joint_steps", "output_dir",
                "scenario_namespace", "checkpoint_seeds"}
    if not isinstance(run_spec, Mapping) or not required <= set(run_spec):
        raise ValueError("Warehouse v6 run spec is missing required fields")
    result = {key: run_spec[key] for key in required}
    smoke = run_spec.get("smoke") is True
    if (not isinstance(result["candidate_id"], str)
            or not result["candidate_id"]
            or "/" in result["candidate_id"] or ".." in result["candidate_id"]
            or not isinstance(result["pair_seed"], int)
            or not isinstance(result["joint_steps"], int)
            or result["joint_steps"] < 120
            or not isinstance(result["scenario_namespace"], str)
            or not result["scenario_namespace"]
            or not isinstance(result["output_dir"], str)
            or not result["output_dir"]):
        raise ValueError("invalid Warehouse v6 run identity or budget")
    seeds = result["checkpoint_seeds"]
    if (not isinstance(seeds, (list, tuple)) or len(seeds) < 2
            or any(not isinstance(value, int) or value < 0 for value in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Warehouse v6 checkpoint seeds must be distinct integers")
    result["checkpoint_seeds"] = list(seeds)
    if not smoke:
        from experiments.warehouse.protocol import (
            BUDGETS, DEVELOPMENT_PAIR_SEEDS, DEVELOPMENT_SELECTION_SEEDS,
            FORMAL_PAIR_SEEDS, FORMAL_SELECTION_SEEDS,
        )
        if result["joint_steps"] not in BUDGETS:
            raise ValueError("Warehouse v6 budget is not preregistered")
        if result["pair_seed"] in DEVELOPMENT_PAIR_SEEDS:
            expected_seeds = DEVELOPMENT_SELECTION_SEEDS
        elif result["pair_seed"] in FORMAL_PAIR_SEEDS:
            expected_seeds = FORMAL_SELECTION_SEEDS
        else:
            raise ValueError("Warehouse v6 pair seed is not preregistered")
        if tuple(seeds) != tuple(expected_seeds):
            raise ValueError("Warehouse v6 checkpoint seeds do not match registered block")
    if smoke:
        result["smoke"] = True
    for key, default in (
        ("bc_episodes", DEFAULT_BC_EPISODES), ("bc_epochs", DEFAULT_BC_EPOCHS),
        ("dagger_rounds", DEFAULT_DAGGER_ROUNDS),
        ("dagger_episodes", DEFAULT_DAGGER_EPISODES),
        ("dagger_epochs", DEFAULT_DAGGER_EPOCHS),
    ):
        value = run_spec.get(key, default)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid Warehouse v6 {key}")
        result[key] = value
    if not result["bc_episodes"] or not result["bc_epochs"]:
        raise ValueError("Warehouse v6 requires offline BC before paired self-play")
    if result["dagger_rounds"] and (not result["dagger_episodes"] or
                                      not result["dagger_epochs"]):
        raise ValueError("Warehouse v6 DAgger rounds need episodes and epochs")
    closure = source_closure_sha256()
    if ("source_closure_sha256" in run_spec
            and run_spec["source_closure_sha256"] != closure):
        raise ValueError("Warehouse v6 run spec source closure mismatch")
    result["source_closure_sha256"] = closure
    return result


def scenario_seed(namespace: str, pair_seed: int, joint_step: int) -> int:
    if joint_step < 0:
        raise ValueError("joint_step must be nonnegative")
    message = f"{namespace}|{pair_seed}|{joint_step // SCENARIO_BLOCK_STEPS}".encode()
    return int.from_bytes(sha256(message).digest()[:8], "big") & 0x7fff_ffff


def scenario_schedule_sha256(namespace: str, pair_seed: int, joint_steps: int) -> str:
    digest = sha256()
    for step in range(joint_steps):
        digest.update(scenario_seed(namespace, pair_seed, step).to_bytes(4, "little"))
    return digest.hexdigest()


def _environment(*, training_reward: bool) -> WarehouseMultiAgentEnv:
    config = participant_study_config(**TRAINING_REWARD_OVERRIDES) if training_reward else \
        participant_study_config()
    return WarehouseMultiAgentEnv(config)


def _set_dual_nn_mode(env: WarehouseMultiAgentEnv) -> None:
    marked = env.get_state()
    marked.participant_controlled_agent_id = "robot_1"
    env.set_state(marked)


def _teacher_rows(episodes: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    observations, masks, labels, summaries = [], [], [], []
    for episode in range(episodes):
        env = _environment(training_reward=False)
        env.reset(seed=seed + episode)
        _set_dual_nn_mode(env)
        while not (env.state.terminated or env.state.truncated):
            obs, mask = encode_public_observation(env.state), action_masks(env.state)
            chosen = public_bc_teacher(obs, mask)
            actions = {role: ACTIONS[int(chosen[index])]
                       for index, role in enumerate(AGENTS)}
            observations.append(obs)
            masks.append(mask)
            labels.append(chosen)
            _, _, _, _, info = env.step(actions)
            if info["proposed_actions"] != actions or info["action_override_agents"]:
                raise RuntimeError("Warehouse offline BC teacher action was rewritten")
        summaries.append((env.state.total_deliveries,
                          env.state.robot_collision_events,
                          bool(env.state.shutdown_count)))
    metrics = np.mean(np.asarray(summaries, dtype=np.float64), axis=0)
    return (torch.from_numpy(np.stack(observations)),
            torch.from_numpy(np.stack(masks)),
            torch.from_numpy(np.stack(labels)).long(),
            {"episodes": episodes, "rows": len(labels),
             "teacher_mean_deliveries": float(metrics[0]),
             "teacher_mean_collisions": float(metrics[1]),
             "teacher_shutdown_episode_rate": float(metrics[2])})


def _bc_epochs(model: WarehouseV6Actor, optimizer: torch.optim.Optimizer,
               obs: torch.Tensor, mask: torch.Tensor, labels: torch.Tensor,
               *, epochs: int, generator: torch.Generator) -> float:
    encoded = policy_features_torch(obs, mask)
    for _ in range(epochs):
        for indices in torch.randperm(len(obs), generator=generator).split(256):
            logits = model.outputs_encoded(encoded[indices], mask[indices])
            loss = nn.functional.cross_entropy(logits.reshape(-1, len(ACTIONS)),
                                               labels[indices].reshape(-1))
            # Both variants inherit the same learned preference for low-bound
            # tree leaves. The raw variant receives no program gradient later.
            loss = loss + 5e-4 * torch.stack([
                model.actors[role].bounds().mean() for role in AGENTS]).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.actors.parameters(), 0.5)
            optimizer.step()
    with torch.no_grad():
        predicted = model.outputs_encoded(encoded, mask).argmax(-1)
        return float((predicted == labels).float().mean())


def pretrain_pair(model: WarehouseV6Actor, *, pair_seed: int, spec: Mapping) -> dict:
    """Shared BC initialization; all labels are a function of public inputs."""
    torch.manual_seed(pair_seed)
    initial_hashes = role_hashes(model)
    obs, mask, labels, teacher = _teacher_rows(spec["bc_episodes"], pair_seed + 10000)
    optimizer = torch.optim.Adam(model.actors.parameters(), lr=3e-4)
    generator = torch.Generator().manual_seed(pair_seed + 31)
    accuracy = _bc_epochs(model, optimizer, obs, mask, labels,
                          epochs=spec["bc_epochs"], generator=generator)
    history = []
    all_obs, all_mask, all_labels = [obs], [mask], [labels]
    sampling = torch.Generator().manual_seed(pair_seed + 41)
    for round_number in range(1, spec["dagger_rounds"] + 1):
        visited_obs, visited_mask, visited_labels, outcomes = [], [], [], []
        for offset in range(spec["dagger_episodes"]):
            seed = pair_seed * 100000 + 70000 + round_number * 1000 + offset
            env = _environment(training_reward=False)
            env.reset(seed=seed)
            _set_dual_nn_mode(env)
            while not (env.state.terminated or env.state.truncated):
                row_obs, row_mask = encode_public_observation(env.state), action_masks(env.state)
                label = public_bc_teacher(row_obs, row_mask)
                with torch.no_grad():
                    logits = model.outputs(torch.from_numpy(row_obs)[None],
                                           torch.from_numpy(row_mask)[None])[0]
                    choice = torch.multinomial(torch.softmax(logits, -1), 1,
                                               generator=sampling).squeeze(-1).numpy()
                actions = {role: ACTIONS[int(choice[index])]
                           for index, role in enumerate(AGENTS)}
                visited_obs.append(row_obs)
                visited_mask.append(row_mask)
                visited_labels.append(label)
                _, _, _, _, info = env.step(actions)
                if info["proposed_actions"] != actions or info["action_override_agents"]:
                    raise RuntimeError("Warehouse offline DAgger NN action was rewritten")
            outcomes.append((env.state.total_deliveries,
                             env.state.robot_collision_events,
                             bool(env.state.shutdown_count)))
        all_obs.append(torch.from_numpy(np.stack(visited_obs)))
        all_mask.append(torch.from_numpy(np.stack(visited_mask)))
        all_labels.append(torch.from_numpy(np.stack(visited_labels)).long())
        joined_obs, joined_mask = torch.cat(all_obs), torch.cat(all_mask)
        joined_labels = torch.cat(all_labels)
        optimizer = torch.optim.Adam(model.actors.parameters(), lr=3e-4)
        accuracy = _bc_epochs(model, optimizer, joined_obs, joined_mask,
                              joined_labels, epochs=spec["dagger_epochs"],
                              generator=generator)
        mean = np.mean(np.asarray(outcomes, dtype=np.float64), axis=0)
        history.append({"round": round_number, "visited_episodes": spec["dagger_episodes"],
                        "visited_joint_states": len(visited_obs),
                        "public_label_accuracy": accuracy,
                        "nn_mean_deliveries": float(mean[0]),
                        "nn_mean_collisions": float(mean[1]),
                        "nn_shutdown_episode_rate": float(mean[2])})
    return {"source": "public_observation_only_offline_bc_and_nn_visits",
            "observation_version": OBSERVATION_VERSION,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "teacher": teacher, "initial_role_sha256": initial_hashes,
            "pretrained_role_sha256": role_hashes(model),
            "public_label_accuracy": accuracy, "dagger": history,
            "teacher_uses_hidden_goal_or_route_state": False}


def collect_episode(model: WarehouseV6Actor, *, seed: int,
                    generator: torch.Generator, max_steps: int):
    env = _environment(training_reward=True)
    env.reset(seed=seed)
    _set_dual_nn_mode(env)
    records = []
    while not (env.state.terminated or env.state.truncated) and len(records) < max_steps:
        obs, mask = encode_public_observation(env.state), action_masks(env.state)
        with torch.no_grad():
            t_obs, t_mask = torch.from_numpy(obs)[None], torch.from_numpy(mask)[None]
            logits = model.outputs(t_obs, t_mask)[0]
            policy = Categorical(logits=logits)
            action = torch.multinomial(policy.probs, 1, generator=generator).squeeze(-1)
            logp = policy.log_prob(action)
            reward_value = model.reward_critic(t_obs.flatten(start_dim=1))[0]
            cost_value = model.cost_critic(t_obs.flatten(start_dim=1))[0, 0]
        commands = {role: ACTIONS[int(action[index])]
                    for index, role in enumerate(AGENTS)}
        previous_shutdowns = env.state.shutdown_count
        _, reward, _, _, info = env.step(commands)
        if info["proposed_actions"] != commands or info["action_override_agents"]:
            raise RuntimeError("Warehouse v6 formal NN action changed before physics")
        records.append({"obs": obs, "mask": mask, "action": action.numpy(),
                        "logp": logp.numpy(), "reward_value": reward_value.numpy().copy(),
                        "cost_value": float(cost_value),
                        "reward": np.asarray([reward[role] for role in AGENTS], dtype=np.float32),
                        "shutdown_cost": float(env.state.shutdown_count > previous_shutdowns),
                        "seed": seed, "frame": env.state.frame - 1})
    complete = bool(env.state.terminated or env.state.truncated)
    if complete:
        reward_bootstrap = np.zeros(len(AGENTS), dtype=np.float32)
        cost_bootstrap = 0.0
    else:
        with torch.no_grad():
            final_obs = torch.from_numpy(encode_public_observation(env.state)).flatten()[None]
            reward_bootstrap = model.reward_critic(final_obs)[0].numpy().copy()
            cost_bootstrap = float(model.cost_critic(final_obs)[0, 0])
    summary = {"seed": seed, "joint_steps": len(records),
               "actor_decisions": len(records) * len(AGENTS),
               "post_policy_interventions": 0,
               "deliveries": env.state.total_deliveries,
               "collisions": env.state.robot_collision_events,
               "shutdown_episode": bool(env.state.shutdown_count),
               "complete": complete}
    return {"records": records, "reward_bootstrap": reward_bootstrap,
            "cost_bootstrap": cost_bootstrap}, summary


def _program_coefficient(progress: float) -> float:
    return PROGRAM_LAMBDA * min(1.0, max(0.0,
        (progress - PROGRAM_START_FRACTION) / PROGRAM_RAMP_FRACTION))


def _program_feedback(logits: torch.Tensor, obs: torch.Tensor, masks: torch.Tensor,
                      sampled_actions: torch.Tensor,
                      program: WarehouseV6Program | None,
                      *, coefficient: float) -> tuple[torch.Tensor, int, float]:
    if program is None or coefficient <= 0.0:
        return logits.new_zeros(()), 0, 0.0
    probabilities = torch.softmax(logits.detach(), -1).cpu().numpy()
    samples = obs.detach().cpu().numpy()
    legal = masks.detach().cpu().numpy()
    sampled = sampled_actions.detach().cpu().numpy()
    targets = probabilities.copy()
    weights = np.zeros(len(samples), dtype=np.float32)
    for index in range(len(samples)):
        actor_action_index = int(probabilities[index].argmax())
        if actor_action_index != int(sampled[index]):
            continue
        row = {"obs": samples[index], "mask": legal[index]}
        evidence = program.covered(row, actor_action=ACTIONS[actor_action_index],
                                   purpose="feedback")
        if evidence is None:
            continue
        distribution = np.asarray([evidence["probabilities"][action] for action in ACTIONS],
                                  dtype=np.float32)
        distribution *= legal[index]
        if float(distribution.sum()) <= 0:
            continue
        distribution /= distribution.sum()
        if int(distribution.argmax()) != actor_action_index:
            continue
        targets[index] = distribution
        weights[index] = 1.0
    count = int(np.count_nonzero(weights))
    if count == 0:
        return logits.new_zeros(()), 0, 0.0
    target_tensor = torch.from_numpy(targets).to(logits)
    weight_tensor = torch.from_numpy(weights).to(logits)
    per_row_kl = torch.sum(target_tensor * (
        torch.log(target_tensor.clamp_min(1e-8)) -
        torch.log_softmax(logits, -1)), dim=-1)
    mean_kl = (per_row_kl * weight_tensor).sum() / weight_tensor.sum()
    return coefficient * mean_kl, count, float(mean_kl.detach())


def _ppo_rows(episodes: Sequence[Mapping]) -> dict[str, torch.Tensor]:
    rows = []
    for episode in episodes:
        reward_adv = np.zeros(len(AGENTS), dtype=np.float32)
        cost_adv = 0.0
        next_reward = episode["reward_bootstrap"]
        next_cost = float(episode["cost_bootstrap"])
        enriched = []
        for item in reversed(episode["records"]):
            reward_delta = item["reward"] + 0.99 * next_reward - item["reward_value"]
            reward_adv = reward_delta + 0.99 * 0.95 * reward_adv
            cost_delta = item["shutdown_cost"] + 0.99 * next_cost - item["cost_value"]
            cost_adv = cost_delta + 0.99 * 0.95 * cost_adv
            enriched.append((item, reward_adv.copy(), reward_adv + item["reward_value"],
                             float(cost_adv), float(cost_adv + item["cost_value"])))
            next_reward = item["reward_value"]
            next_cost = item["cost_value"]
        rows.extend(reversed(enriched))
    if not rows:
        raise RuntimeError("Warehouse v6 PPO batch has no transitions")
    return {
        "obs": torch.from_numpy(np.stack([row[0]["obs"] for row in rows])),
        "mask": torch.from_numpy(np.stack([row[0]["mask"] for row in rows])),
        "actions": torch.from_numpy(np.stack([row[0]["action"] for row in rows])).long(),
        "old_logp": torch.from_numpy(np.stack([row[0]["logp"] for row in rows])),
        "reward_adv": torch.from_numpy(np.stack([row[1] for row in rows])),
        "reward_target": torch.from_numpy(np.stack([row[2] for row in rows])),
        "cost_adv": torch.tensor([row[3] for row in rows], dtype=torch.float32),
        "cost_target": torch.tensor([row[4] for row in rows], dtype=torch.float32),
    }


def ppo_update(model: WarehouseV6Actor, optimizer: torch.optim.Optimizer,
               episodes: Sequence[Mapping], programs: Mapping[str, WarehouseV6Program | None] | None,
               *, progress: float, cost_multiplier: float, epochs: int = 4) -> dict:
    data = _ppo_rows(episodes)
    obs, masks, actions, old_logp = (data[key] for key in
                                     ("obs", "mask", "actions", "old_logp"))
    encoded = policy_features_torch(obs, masks)
    reward_adv = data["reward_adv"]
    reward_adv = ((reward_adv - reward_adv.mean(0, keepdim=True)) /
                  reward_adv.std(0, keepdim=True, unbiased=False).clamp_min(1e-6))
    cost_adv = data["cost_adv"]
    cost_adv = ((cost_adv - cost_adv.mean()) /
                cost_adv.std(unbiased=False).clamp_min(1e-6))
    coefficient = _program_coefficient(progress) if programs is not None else 0.0
    counts = {role: 0 for role in AGENTS}
    gradient_norms = {role: 0.0 for role in AGENTS}
    exposure = {role: 0 for role in AGENTS}
    zero_coverage = {role: 0 for role in AGENTS}
    no_legal = {role: 0 for role in AGENTS}
    program_kl = {role: 0.0 for role in AGENTS}
    losses = []
    for _ in range(epochs):
        logits = model.outputs_encoded(encoded, masks)
        policy = Categorical(logits=logits)
        ratio = torch.exp(policy.log_prob(actions) - old_logp)
        reward_surrogate = torch.minimum(
            ratio * reward_adv, ratio.clamp(0.8, 1.2) * reward_adv)
        # The shutdown cost is an observed episode event, not a hand-written
        # action-risk matrix or a run-time safety controller.
        cost_surrogate = torch.maximum(
            ratio * cost_adv[:, None], ratio.clamp(0.8, 1.2) * cost_adv[:, None])
        reward_value = model.reward_critic(obs.flatten(start_dim=1))
        cost_value = model.cost_critic(obs.flatten(start_dim=1)).squeeze(-1)
        loss = (-reward_surrogate.mean() + cost_multiplier * cost_surrogate.mean()
                + 0.5 * nn.functional.mse_loss(reward_value, data["reward_target"])
                + 0.25 * nn.functional.mse_loss(cost_value, data["cost_target"])
                - 0.005 * policy.entropy().mean()
                + 5e-4 * torch.stack([
                    model.actors[role].bounds().mean() for role in AGENTS]).mean())
        if programs is not None:
            for index, role in enumerate(AGENTS):
                penalty, count, kl = _program_feedback(
                    logits[:, index], obs[:, index], masks[:, index], actions[:, index],
                    programs.get(role), coefficient=coefficient)
                if count:
                    grads = torch.autograd.grad(
                        penalty, tuple(model.actors[role].parameters()),
                        retain_graph=True, allow_unused=True)
                    gradient_norms[role] += float(sum(
                        grad.detach().square().sum().item()
                        for grad in grads if grad is not None) ** 0.5)
                loss = loss + penalty
                counts[role] += count
                program_kl[role] += kl
                if coefficient > 0:
                    exposure[role] += len(obs)
                    zero_coverage[role] += len(obs) - count
                    no_legal[role] += int((masks[:, index].sum(-1) <= 0.5).sum())
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        losses.append(float(loss.detach()))
    return {"loss": float(np.mean(losses)), "cost_multiplier": cost_multiplier,
            "program_coefficient": coefficient, "feedback_rows": counts,
            "feedback_gradient_norm": gradient_norms, "program_kl": program_kl,
            "feedback_window_actor_rows": exposure,
            "zero_action_coverage_rows": zero_coverage,
            "physical_no_legal_action_rows": no_legal,
            "batch_shutdown_episode_rate": float(np.mean([
                summary["records"][-1]["shutdown_cost"] > 0
                if summary["records"] else False for summary in episodes])),
            "cost_critic_target_mean": float(data["cost_target"].mean())}


def evaluate(model: WarehouseV6Actor | FrozenWarehouseV6Actor,
             seeds: Sequence[int], *, action_seed_offset: int = 0) -> dict:
    frozen = (FrozenWarehouseV6Actor.from_model(model)
              if isinstance(model, WarehouseV6Actor) else model)
    episodes = []
    for seed in seeds:
        env = _environment(training_reward=False)
        env.reset(seed=int(seed))
        _set_dual_nn_mode(env)
        while not (env.state.terminated or env.state.truncated):
            actions, _ = frozen.act(env.state, base_seed=int(seed) + action_seed_offset)
            _, _, _, _, info = env.step(actions)
            if info["proposed_actions"] != actions or info["action_override_agents"]:
                raise RuntimeError("Warehouse v6 evaluator rewrote a frozen NN action")
        episodes.append({"seed": int(seed), "deliveries": env.state.total_deliveries,
                         "collisions": env.state.robot_collision_events,
                         "shutdown_episode": bool(env.state.shutdown_count)})
    summary = {
        "mean_deliveries": float(np.mean([row["deliveries"] for row in episodes])),
        "mean_collisions": float(np.mean([row["collisions"] for row in episodes])),
        "shutdown_episode_rate": float(np.mean([row["shutdown_episode"] for row in episodes])),
    }
    summary["quality"] = (summary["mean_deliveries"]
                          - 0.25 * summary["mean_collisions"]
                          - 5.0 * summary["shutdown_episode_rate"])
    summary["formal_quality_passed"] = bool(
        summary["mean_deliveries"] >= 5.0 and summary["mean_collisions"] <= 8.0
        and summary["shutdown_episode_rate"] <= 0.10)
    summary["development_quality_passed"] = bool(
        summary["mean_deliveries"] >= 5.5 and summary["mean_collisions"] <= 6.0
        and summary["shutdown_episode_rate"] <= 0.05)
    return {"summary": summary, "episodes": episodes}


def _program_rows(episodes: Sequence[Mapping], role_index: int) -> list[dict]:
    return [{"obs": record["obs"][role_index].copy(),
             "mask": record["mask"][role_index].copy(),
             "episode": str(record["seed"])}
            for episode in episodes for record in episode["records"]]


def _fit_training_programs(model: WarehouseV6Actor,
                           buffers: Mapping[str, deque]) -> tuple[dict, dict]:
    actor = FrozenWarehouseV6Actor.from_model(model)
    programs, reports = {}, {}
    split_names = ("fit", "calibration", "eligibility", "validation")
    for role in AGENTS:
        rows = list(buffers[role])
        episodes = list(dict.fromkeys(row["episode"] for row in rows))[-PROGRAM_MIN_EPISODES:]
        if len(episodes) < PROGRAM_MIN_EPISODES:
            programs[role] = None
            reports[role] = {"error": "insufficient distinct training scenario episodes",
                             "episodes": len(episodes)}
            continue
        groups, start = {}, 0
        for name, count in zip(split_names, PROGRAM_SPLIT_EPISODES, strict=True):
            groups[name] = set(episodes[start:start + count])
            start += count
        parts = {name: [row for row in rows if row["episode"] in group]
                 for name, group in groups.items()}
        try:
            program = fit_program(role, actor, *(parts[name] for name in split_names))
            programs[role] = program
            reports[role] = {"program_sha256": program.sha256(),
                             "actor_role_sha256": actor.role_sha256[role],
                             "training_split_episode_counts": {
                                 name: len(group) for name, group in groups.items()},
                             "validation": program.report["validation"]}
        except (ValueError, RuntimeError) as exc:
            programs[role] = None
            reports[role] = {"error": f"{type(exc).__name__}: {exc}"}
    return programs, reports


def collect_program_audit_rows(actor: FrozenWarehouseV6Actor,
                               seeds: Sequence[int]) -> dict[str, list[dict]]:
    """Independent neural rollouts; no program call can change their actions."""
    rows = {role: [] for role in AGENTS}
    for seed in seeds:
        env = _environment(training_reward=False)
        env.reset(seed=int(seed))
        _set_dual_nn_mode(env)
        while not (env.state.terminated or env.state.truncated):
            obs, masks = encode_public_observation(env.state), action_masks(env.state)
            for index, role in enumerate(AGENTS):
                rows[role].append({"obs": obs[index].copy(),
                                   "mask": masks[index].copy(),
                                   "episode": str(seed)})
            actions, _ = actor.act(env.state, base_seed=int(seed))
            _, _, _, _, info = env.step(actions)
            if info["proposed_actions"] != actions or info["action_override_agents"]:
                raise RuntimeError("Warehouse v6 audit rollout rewrote an NN action")
    return rows


def audit_programs(actor: FrozenWarehouseV6Actor,
                   programs: Mapping[str, WarehouseV6Program],
                   seeds: Sequence[int]) -> dict[str, dict]:
    if set(programs) != set(AGENTS):
        raise ValueError("Warehouse v6 audit requires both role programs")
    samples = collect_program_audit_rows(actor, seeds)
    return {role: audit_frozen_program(programs[role], actor, samples[role])
            for role in AGENTS}


def _role_weight_change(initial: Mapping[str, torch.Tensor],
                        model: WarehouseV6Actor) -> dict[str, float]:
    selected = model.state_dict()
    result = {}
    for role in AGENTS:
        prefix = f"actors.{role}."
        result[role] = float(sum(
            (selected[name].detach().cpu().double() - before.double()).square().sum().item()
            for name, before in initial.items() if name.startswith(prefix)) ** 0.5)
    return result


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _variant_checkpoint_path(directory: Path, variant: str) -> Path:
    return directory / f"{variant}.checkpoint.pt"


def _train_variant(initial: Mapping[str, torch.Tensor], *, variant: str,
                   spec: Mapping, directory: Path) -> tuple[WarehouseV6Actor, dict, dict]:
    if variant not in ("raw", "regularized"):
        raise ValueError("invalid Warehouse v6 training variant")
    model = WarehouseV6Actor()
    model.load_state_dict(initial)
    initial_role_hashes = role_hashes(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    generator = torch.Generator().manual_seed(spec["pair_seed"] + 1)
    checkpoint = _variant_checkpoint_path(directory, variant)
    spec_hash = _digest_json({key: value for key, value in spec.items()
                              if key != "output_dir"})
    used_steps, episode_number = 0, 0
    histories, updates, development = [], [], []
    buffers = {role: deque(maxlen=PROGRAM_BUFFER_STATES) for role in AGENTS}
    programs = {role: None for role in AGENTS} if variant == "regularized" else None
    extractions = {role: [] for role in AGENTS}
    best = None
    cost_multiplier = 1.0
    interval = max(1, spec["joint_steps"] // 10)
    next_eval = interval
    if checkpoint.is_file():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (saved.get("spec_sha256") != spec_hash
                or saved.get("source_closure_sha256") != spec["source_closure_sha256"]
                or saved.get("variant") != variant
                or saved.get("initial_role_sha256") != initial_role_hashes
                or saved.get("model_version") != MODEL_VERSION
                or saved.get("feature_schema_sha256") != FEATURE_SCHEMA_SHA256):
            raise RuntimeError("Warehouse v6 checkpoint identity or Actor start mismatch")
        if saved.get("complete"):
            model.load_state_dict(saved["selected_state_dict"])
            restored = {role: WarehouseV6Program.from_dict(
                payload, actor_role_sha256=role_hashes(model)[role])
                for role, payload in saved.get("selected_programs", {}).items()}
            return model, restored, saved["report"]
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["generator_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        used_steps, episode_number = saved["used_steps"], saved["episode_number"]
        histories, updates, development = (saved[key] for key in
                                            ("histories", "updates", "development"))
        best, cost_multiplier, next_eval = (saved[key] for key in
                                            ("best", "cost_multiplier", "next_eval"))
        if programs is not None:
            for role in AGENTS:
                buffers[role].extend(saved["buffers"][role])
                payload = saved["programs"][role]
                programs[role] = (WarehouseV6Program.from_dict(
                    payload, actor_role_sha256=payload["actor_role_sha256"])
                    if payload is not None else None)
            extractions = saved["extractions"]
    while used_steps < spec["joint_steps"]:
        batch, batch_summaries = [], []
        while used_steps < spec["joint_steps"] and len(batch) < 16:
            seed = scenario_seed(spec["scenario_namespace"], spec["pair_seed"], used_steps)
            block_remaining = SCENARIO_BLOCK_STEPS - used_steps % SCENARIO_BLOCK_STEPS
            episode, summary = collect_episode(
                model, seed=seed, generator=generator,
                max_steps=min(block_remaining, spec["joint_steps"] - used_steps))
            if not summary["joint_steps"]:
                raise RuntimeError("Warehouse v6 self-play rollout yielded zero steps")
            batch.append(episode)
            batch_summaries.append(summary)
            histories.append(summary)
            used_steps += summary["joint_steps"]
            episode_number += 1
            if programs is not None:
                for index, role in enumerate(AGENTS):
                    buffers[role].extend(_program_rows((episode,), index))
        update = ppo_update(model, optimizer, batch, programs,
                            progress=used_steps / spec["joint_steps"],
                            cost_multiplier=cost_multiplier)
        # The dual variable sees actual completed episode outcomes. Segment
        # endings are 120-step evaluation episodes, including early shutdowns.
        complete_outcomes = [row["shutdown_episode"] for row in batch_summaries
                             if row["complete"]]
        shutdown_rate = (float(np.mean(complete_outcomes))
                         if complete_outcomes else None)
        if shutdown_rate is not None:
            cost_multiplier = min(20.0, max(0.0,
                cost_multiplier + 0.5 * (shutdown_rate - 0.04)))
        update.update(joint_steps=used_steps, complete_episode_shutdown_rate=shutdown_rate,
                      completed_episodes_for_dual=len(complete_outcomes),
                      next_cost_multiplier=cost_multiplier)
        updates.append(update)
        if used_steps >= next_eval or used_steps == spec["joint_steps"]:
            _assert_source_closure(spec["source_closure_sha256"])
            if programs is not None and used_steps / spec["joint_steps"] >= PROGRAM_START_FRACTION:
                programs, extracted = _fit_training_programs(model, buffers)
                for role in AGENTS:
                    extractions[role].append({"joint_steps": used_steps, **extracted[role]})
            measured = evaluate(model, spec["checkpoint_seeds"])
            checkpoint_result = {"joint_steps": used_steps, **measured["summary"]}
            development.append(checkpoint_result)
            selectable = variant == "raw" or used_steps / spec["joint_steps"] >= 0.85
            from experiments.warehouse.protocol import DEVELOPMENT_PAIR_SEEDS
            threshold = ("development_quality_passed"
                         if spec["pair_seed"] in DEVELOPMENT_PAIR_SEEDS else
                         "formal_quality_passed")
            rank = (bool(checkpoint_result[threshold]),
                    float(checkpoint_result["quality"]))
            if selectable and (best is None or rank > best["rank"]):
                best = {"state_dict": deepcopy(model.state_dict()),
                        "rank": rank, "development": checkpoint_result,
                        "programs": ({role: deepcopy(programs[role])
                                      for role in AGENTS} if programs is not None else {}),
                        "selected_update_index": len(updates)}
            next_eval += interval
            _save_checkpoint(checkpoint, {
                "spec_sha256": spec_hash, "variant": variant,
                "source_closure_sha256": spec["source_closure_sha256"],
                "initial_role_sha256": initial_role_hashes,
                "model_version": MODEL_VERSION,
                "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
                "complete": False, "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "generator_state": generator.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "used_steps": used_steps, "episode_number": episode_number,
                "histories": histories, "updates": updates,
                "development": development, "best": best,
                "cost_multiplier": cost_multiplier, "next_eval": next_eval,
                "buffers": ({role: list(buffers[role]) for role in AGENTS}
                            if programs is not None else {}),
                "programs": ({role: (programs[role].to_dict()
                                      if programs[role] is not None else None)
                              for role in AGENTS} if programs is not None else {}),
                "extractions": extractions,
            })
    if best is None:
        raise RuntimeError("Warehouse v6 training produced no selected checkpoint")
    _assert_source_closure(spec["source_closure_sha256"])
    model.load_state_dict(best["state_dict"])
    selected_programs = {role: program for role, program in best["programs"].items()
                         if program is not None}
    if variant == "regularized" and selected_programs:
        current_hashes = role_hashes(model)
        if any(selected_programs[role].actor_role_sha256 != current_hashes[role]
               for role in selected_programs):
            raise RuntimeError("Warehouse v6 selected program does not bind selected Actor")
    selected_updates = updates[:best["selected_update_index"]]
    feedback_rows = {role: sum(item["feedback_rows"][role] for item in selected_updates)
                     for role in AGENTS}
    gradient_norm = {role: sum(item["feedback_gradient_norm"][role]
                               for item in selected_updates) for role in AGENTS}
    report = {
        "runner_version": RUNNER_VERSION, "candidate_id": spec["candidate_id"],
        "source_closure_sha256": spec["source_closure_sha256"],
        "pair_seed": spec["pair_seed"], "variant": variant,
        "joint_steps": used_steps, "actor_decisions": 2 * used_steps,
        "model_version": MODEL_VERSION, "actor_model_version": MODEL_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "run_spec_sha256": spec_hash,
        "checkpoint_seeds": list(spec["checkpoint_seeds"]),
        "checkpoint_seed_sha256": _digest_json(spec["checkpoint_seeds"]),
        "scenario_schedule_sha256": scenario_schedule_sha256(
            spec["scenario_namespace"], spec["pair_seed"], spec["joint_steps"]),
        "initial_role_sha256": initial_role_hashes,
        "selected_role_sha256": role_hashes(model),
        "role_weight_change_norm": _role_weight_change(initial, model),
        "post_policy_interventions": 0,
        "physical_no_legal_action_rows": sum(
            sum(item["physical_no_legal_action_rows"].values()) for item in updates),
        "feedback_rows": feedback_rows,
        "feedback_gradient_norm": gradient_norm,
        "feedback_window_actor_rows": {role: sum(
            item["feedback_window_actor_rows"][role] for item in selected_updates)
            for role in AGENTS},
        "zero_action_coverage_rows": {role: sum(
            item["zero_action_coverage_rows"][role] for item in selected_updates)
            for role in AGENTS},
        "cost_critic_training": True, "cost_target_event": "actual_episode_shutdown",
        "cost_constraint_target": 0.04, "final_cost_multiplier": cost_multiplier,
        "training_reward_overrides": dict(TRAINING_REWARD_OVERRIDES),
        "selected_development": best["development"],
        "quality": {key: value for key, value in best["development"].items()
                    if key != "joint_steps"},
        "development": development,
        "training_episodes": histories,
        "updates": updates,
        "program_extractions": extractions,
        "selected_program_reports": {role: program.report
                                     for role, program in selected_programs.items()},
    }
    if variant == "raw" and (any(feedback_rows.values()) or any(gradient_norm.values())):
        raise RuntimeError("Warehouse v6 raw baseline received a program gradient")
    _save_checkpoint(checkpoint, {
        "spec_sha256": spec_hash, "variant": variant,
        "source_closure_sha256": spec["source_closure_sha256"],
        "initial_role_sha256": initial_role_hashes,
        "model_version": MODEL_VERSION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "complete": True, "selected_state_dict": model.state_dict(),
        "selected_programs": {role: program.to_dict()
                              for role, program in selected_programs.items()},
        "report": report,
    })
    return model, selected_programs, report


def train_pair(run_spec: Mapping) -> dict:
    """Train both variants from one BC start; write isolated, resumable runs."""
    spec = _checked_spec(run_spec)
    directory = (Path(spec["output_dir"]) / spec["candidate_id"] /
                 f"pair_{spec['pair_seed']}_steps_{spec['joint_steps']}")
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(spec["pair_seed"])
    base = WarehouseV6Actor()
    bc_checkpoint = directory / "paired_bc.checkpoint.pt"
    spec_hash = _digest_json({key: value for key, value in spec.items()
                              if key != "output_dir"})
    if bc_checkpoint.is_file():
        saved = torch.load(bc_checkpoint, map_location="cpu", weights_only=False)
        if (saved.get("spec_sha256") != spec_hash
                or saved.get("source_closure_sha256") != spec["source_closure_sha256"]):
            raise RuntimeError("Warehouse v6 BC checkpoint run spec mismatch")
        base.load_state_dict(saved["state_dict"])
        bc_report = saved["report"]
        if bc_report["pretrained_role_sha256"] != role_hashes(base):
            raise RuntimeError("Warehouse v6 BC checkpoint Actor hashes mismatch")
    else:
        bc_report = pretrain_pair(base, pair_seed=spec["pair_seed"], spec=spec)
        _assert_source_closure(spec["source_closure_sha256"])
        _save_checkpoint(bc_checkpoint, {"spec_sha256": spec_hash,
                                         "source_closure_sha256": spec["source_closure_sha256"],
                                         "state_dict": base.state_dict(),
                                         "report": bc_report})
    initial = deepcopy(base.state_dict())
    variants = {}
    for variant in ("raw", "regularized"):
        _assert_source_closure(spec["source_closure_sha256"])
        model, programs, report = _train_variant(
            initial, variant=variant, spec=spec, directory=directory)
        actor_path = directory / f"{variant}.actor.npz"
        actor_artifact = export_actor(model, actor_path,
                                      pair_seed=spec["pair_seed"],
                                      variant=variant, candidate_id=spec["candidate_id"])
        program_artifacts = {}
        for role, program in programs.items():
            path = directory / f"{variant}.{role}.program.json"
            digest = program.save(path)
            program_artifacts[role] = {"path": str(path), "sha256": digest,
                                       "actor_role_sha256": program.actor_role_sha256}
        report["actor_artifact"] = actor_artifact
        report["program_artifacts"] = program_artifacts
        report_path = directory / f"{variant}.training.json"
        report_path.write_text(json.dumps(report, sort_keys=True,
                                          separators=(",", ":"), allow_nan=False))
        report["training_report_path"] = str(report_path)
        report["training_report_sha256"] = sha256(report_path.read_bytes()).hexdigest()
        variants[variant] = report
    if variants["raw"]["initial_role_sha256"] != variants["regularized"]["initial_role_sha256"]:
        raise RuntimeError("Warehouse v6 paired Actors did not share BC initialization")
    if variants["raw"]["scenario_schedule_sha256"] != \
            variants["regularized"]["scenario_schedule_sha256"]:
        raise RuntimeError("Warehouse v6 paired variants used different scenarios")
    _assert_source_closure(spec["source_closure_sha256"])
    pair_report = {
        "candidate_id": spec["candidate_id"], "pair_seed": spec["pair_seed"],
        "joint_steps": spec["joint_steps"], "run_spec_sha256": spec_hash,
        "source_closure_sha256": spec["source_closure_sha256"],
        "source_files_sha256": source_closure_files_sha256(),
        "checkpoint_seeds": list(spec["checkpoint_seeds"]),
        "bc": bc_report, "initial_role_sha256": role_hashes(base),
        "variants": variants,
    }
    (directory / "pair.training.json").write_text(json.dumps(
        pair_report, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return pair_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Warehouse v6 paired dual-Actor run")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--pair-seed", type=int, required=True)
    parser.add_argument("--joint-steps", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scenario-namespace", default="warehouse-v6-matched-120-step-blocks")
    parser.add_argument("--checkpoint-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--bc-episodes", type=int, default=DEFAULT_BC_EPISODES)
    parser.add_argument("--bc-epochs", type=int, default=DEFAULT_BC_EPOCHS)
    parser.add_argument("--dagger-rounds", type=int, default=DEFAULT_DAGGER_ROUNDS)
    parser.add_argument("--dagger-episodes", type=int, default=DEFAULT_DAGGER_EPISODES)
    parser.add_argument("--dagger-epochs", type=int, default=DEFAULT_DAGGER_EPOCHS)
    parser.add_argument("--smoke", action="store_true",
                        help="allow a tiny non-protocol test run")
    args = parser.parse_args()
    report = train_pair(vars(args))
    print(json.dumps({"candidate_id": report["candidate_id"],
                      "pair_seed": report["pair_seed"],
                      "joint_steps": report["joint_steps"],
                      "variants": {
                          variant: {"quality": row["quality"],
                                    "feedback_rows": row["feedback_rows"],
                                    "feedback_gradient_norm": row["feedback_gradient_norm"]}
                          for variant, row in report["variants"].items()}},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
