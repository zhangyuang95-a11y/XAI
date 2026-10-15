"""Matched Warehouse v5 dual-Actor PPO with optional in-loop RCPD.

Run the full command without reduced budgets before considering these actors
for the study. The script never turns a failed development gate into a release.
"""
from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from backend.training.warehouse_pure_nn import teacher_rows
from core.rcpd import RCPD
from backend.training.warehouse_selective_v5 import (
    SelectiveLeafProgram, audit_frozen_program, fit_selective_program,
)
from env.warehouse.coordination import stable_coordination_actions
from env.warehouse.domain import participant_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.layouts import STUDY_MAP_LAYOUT
from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import (
    AGENTS, DUAL_MODEL_VERSION, OBSERVATION_DIM, OBSERVATION_VERSION,
    FrozenDualPureNN, FrozenPureNN, action_masks, observation_feature_names,
    observations, role_weight_sha256,
)

ROOT = Path(__file__).resolve().parents[4]
V5_CONFIG = ROOT / "configs/study_v3_warehouse_neural_v5.json"
DEFAULT_SEEDS = (6001, 6002, 6003)
FULL_JOINT_STEPS = 200_000
SCENARIO_BLOCK_STEPS = 120
SCENARIO_SCHEDULE_VERSION = "matched-120-step-blocks-participant-reward.v2"
TRAINING_REWARD_VERSION = "participant-score-v4-role-credit-collision-minus100-shutdown-minus200"
TRAINING_REWARD_OVERRIDES = {"robot_collision_points": -100.0,
                             "shutdown_points": -200.0}
PROGRAM_FEEDBACK_VERSION = "selective-rcpd-two-split-leaf-final-independent.v5.3"
PROGRAM_LAMBDA = 0.01
PROGRAM_START_FRACTION = 0.80
PROGRAM_RAMP_FRACTION = 0.05
PROGRAM_EXTRACTION_INTERVAL = 20_000
PROGRAM_BUFFER_STATES = 20_000
PROGRAM_MIN_EPISODES = 124
PROGRAM_SPLIT_EPISODES = (50, 25, 25, 24)
PROGRAM_HARD_WEIGHT = {"robot_1": 0.5, "robot_2": 0.0}
ZERO_ACTION_COVERAGE_DEFINITION = (
    "PPO role-row presentations during positive program coefficient, "
    "counting every optimizer epoch; no validated, legal program "
    "action agreed with both current masked NN argmax and the "
    "sampled executed NN action")
BC_ARCHITECTURE_VERSION = "dual-actor-role-critic-offline-dagger-v3"
DAGGER_ROUNDS = 6
DAGGER_VISITED_EPISODES_PER_ROUND = 80
DAGGER_ORIGINAL_TEACHER_EPISODES = 100
DAGGER_EPOCHS_PER_ROUND = 20
# A development-only perturbation of pair 6001's offline relabel states found
# that environment-owned goal history affects teacher labels even though the
# raw Actor observation omits that history. This is an offline BC limitation,
# not an action source during formal NN–NN PPO or frozen execution.
BC_TEACHER_OBSERVABILITY_PROBE = {
    "scope": "pair_6001_bc_nn_visits_training_seeds_90000_through_90019",
    "joint_pre_action_states": 2400,
    "cleared_goal_and_route_memory_changed_teacher_joint_action": 209,
    "also_cleared_charging_history_changed_teacher_joint_action": 292,
    "limitation": ("Offline teacher labels use the pre-action environment state, "
                   "including persistent goals absent from the NN observation; "
                   "the formal self-play and execution actions remain NN-selected."),
}
DEVELOPMENT_SEEDS = tuple(range(240000, 240024))
FINAL_PROGRAM_AUDIT_SEEDS = tuple(range(240100, 240124))
HELDOUT_BASE_SEEDS = tuple(range(250000, 250100))
HELDOUT_SEEDS_BY_TASK = {
    task: tuple(seed * 10 + task for seed in HELDOUT_BASE_SEEDS)
    for task in (1, 2, 3)
}
FEATURE_NAMES = observation_feature_names()


def _actor_network(hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(OBSERVATION_DIM, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, len(ACTIONS)))


class DualNN(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.actors = nn.ModuleDict({agent_id: _actor_network(hidden) for agent_id in AGENTS})
        # The environment supplies shared delivery/collision credit plus
        # role-specific mission progress and regret.  Preserve that credit in
        # a centralized value estimate for each Actor rather than averaging
        # the two rewards into an identical advantage.
        self.critic = nn.Sequential(nn.Linear(2 * OBSERVATION_DIM, hidden), nn.ReLU(),
                                    nn.Linear(hidden, hidden), nn.ReLU(),
                                    nn.Linear(hidden, len(AGENTS)))

    def outputs(self, features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        if features.shape[-2:] != (2, OBSERVATION_DIM):
            raise ValueError("dual Actor input must contain both role observations")
        result = torch.stack([self.actors[agent_id](features[..., row, :])
                              for row, agent_id in enumerate(AGENTS)], dim=-2)
        return result.masked_fill(masks <= 0.5, -1.0e9)


def _role_weights(model: DualNN, agent_id: str) -> dict[str, np.ndarray]:
    layers = (model.actors[agent_id][0], model.actors[agent_id][2], model.actors[agent_id][4])
    result = {}
    for index, layer in enumerate(layers):
        result[f"w{index}"] = layer.weight.detach().cpu().numpy().astype(np.float32).copy()
        result[f"b{index}"] = layer.bias.detach().cpu().numpy().astype(np.float32).copy()
    return result


def _frozen(model: DualNN) -> FrozenDualPureNN:
    return FrozenDualPureNN({agent_id: FrozenPureNN(_role_weights(model, agent_id), "in-memory")
                             for agent_id in AGENTS}, "in-memory")


def _weight_hashes(model: DualNN) -> dict[str, str]:
    return {agent_id: role_weight_sha256(_role_weights(model, agent_id)) for agent_id in AGENTS}


def _actor_weight_changes(initial: dict, model: DualNN) -> tuple[dict[str, float], dict[str, int]]:
    """Measure each selected Actor against its paired offline BC start."""
    selected = model.state_dict()
    norms, changed = {}, {}
    for role in AGENTS:
        prefix = f"actors.{role}."
        squared = 0.0
        count = 0
        for name, before in initial.items():
            if name.startswith(prefix):
                delta = selected[name].detach().cpu().double() - before.detach().cpu().double()
                squared += float(torch.sum(delta.square()))
                count += int(torch.count_nonzero(delta))
        norms[role] = squared ** 0.5
        changed[role] = count
    return norms, changed


def _scenario_seed(pair_seed: int, joint_step: int) -> int:
    """Assign a common scenario to each fixed 120-step budget block.

    A shutdown can end one policy's episode early. It may reset inside the
    same block, but it cannot advance that variant to a new scenario earlier
    than its paired run.
    """
    return 1_000_000 + pair_seed * 10_000 + joint_step // SCENARIO_BLOCK_STEPS


def _scenario_schedule_sha256(pair_seed: int, joint_steps: int) -> str:
    digest = sha256()
    for step in range(joint_steps):
        digest.update(_scenario_seed(pair_seed, step).to_bytes(8, "little"))
    return digest.hexdigest()


def _validate_scenario_schedule(history: list[dict], pair_seed: int,
                                joint_steps: int) -> None:
    scheduled_steps = 0
    for episode in history:
        if (episode["seed"] != _scenario_seed(pair_seed, scheduled_steps)
                or episode["joint_steps"] < 1
                or episode["joint_steps"] >
                SCENARIO_BLOCK_STEPS - scheduled_steps % SCENARIO_BLOCK_STEPS):
            raise RuntimeError("Warehouse paired scenario schedule drifted")
        scheduled_steps += episode["joint_steps"]
    if scheduled_steps != joint_steps:
        raise RuntimeError("Warehouse scenario schedule has wrong step count")


def pretrain_pair(model: DualNN, *, seed: int, episodes: int, epochs: int) -> dict:
    """Offline teacher and relabeled NN visits initialize both PPO variants.

    Teacher labels are collected only before formal self-play.  The sampled
    NN joint action, never the offline label, advances each visited rollout.
    A shadow environment recomputes any public coordination memory from the
    current pre-action state before assigning a label.
    """
    features, masks, labels = teacher_rows(episodes, seed + 10000)
    initial = _weight_hashes(model)
    optimizer = torch.optim.Adam(model.actors.parameters(), lr=3e-4)
    generator = torch.Generator().manual_seed(seed + 31)
    for _ in range(epochs):
        for indices in torch.randperm(len(labels) // 2, generator=generator).split(256):
            x = features.reshape(-1, 2, OBSERVATION_DIM)[indices]
            m = masks.reshape(-1, 2, len(ACTIONS))[indices]
            y = labels.reshape(-1, 2)[indices]
            loss = nn.functional.cross_entropy(model.outputs(x, m).reshape(-1, len(ACTIONS)), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        correct = (model.outputs(features.reshape(-1, 2, OBSERVATION_DIM),
                                 masks.reshape(-1, 2, len(ACTIONS))).argmax(-1)
                   == labels.reshape(-1, 2)).float().mean()
    initial_teacher_accuracy = float(correct)

    original_x, original_m, original_y = teacher_rows(
        DAGGER_ORIGINAL_TEACHER_EPISODES, seed + 10000)
    all_x = [original_x.reshape(-1, 2, OBSERVATION_DIM)]
    all_m = [original_m.reshape(-1, 2, len(ACTIONS))]
    all_y = [original_y.reshape(-1, 2)]
    sampling_generator = torch.Generator().manual_seed(seed + 1)
    rounds = []
    for round_number in range(1, DAGGER_ROUNDS + 1):
        visited_x, visited_m, visited_y = [], [], []
        action_agreement, rollout_results = [], []
        scenario_seeds = range(70000 + round_number * 100,
                               70000 + round_number * 100 +
                               DAGGER_VISITED_EPISODES_PER_ROUND)
        for scenario_seed in scenario_seeds:
            env = WarehouseMultiAgentEnv(participant_study_config())
            env.reset(seed=scenario_seed)
            marked = env.get_state()
            marked.participant_controlled_agent_id = "robot_1"
            env.set_state(marked)
            shadow = WarehouseMultiAgentEnv(participant_study_config())
            shadow.reset(seed=scenario_seed)
            while not (env.state.terminated or env.state.truncated):
                obs, mask = observations(env.state), action_masks(env.state)
                shadow.set_state(env.get_state())
                teacher = stable_coordination_actions(shadow)
                label = np.asarray([ACTIONS.index(teacher[role]) for role in AGENTS],
                                   dtype=np.int64)
                with torch.no_grad():
                    logits = model.outputs(torch.from_numpy(obs)[None],
                                           torch.from_numpy(mask)[None])[0]
                    indices = torch.multinomial(torch.softmax(logits, -1), 1,
                                                generator=sampling_generator).squeeze(-1)
                sampled = indices.numpy()
                action_agreement.append(float(np.mean(sampled == label)))
                actions = {role: ACTIONS[int(sampled[row])]
                           for row, role in enumerate(AGENTS)}
                visited_x.append(obs)
                visited_m.append(mask)
                visited_y.append(label)
                _, _, _, _, info = env.step(actions)
                if info["proposed_actions"] != actions or info["action_override_agents"]:
                    raise RuntimeError("offline Warehouse NN visit had a post-policy action override")
            rollout_results.append((env.state.total_deliveries,
                                    env.state.robot_collision_events,
                                    bool(env.state.shutdown_count)))
        all_x.append(torch.as_tensor(np.asarray(visited_x)))
        all_m.append(torch.as_tensor(np.asarray(visited_m)))
        all_y.append(torch.as_tensor(np.asarray(visited_y)))
        x, mask, y = torch.cat(all_x), torch.cat(all_m), torch.cat(all_y)
        round_optimizer = torch.optim.Adam(model.actors.parameters(), lr=3e-4)
        round_generator = torch.Generator().manual_seed(91000 + round_number)
        for _ in range(DAGGER_EPOCHS_PER_ROUND):
            for indices in torch.randperm(len(x), generator=round_generator).split(256):
                loss = nn.functional.cross_entropy(
                    model.outputs(x[indices], mask[indices]).reshape(-1, len(ACTIONS)),
                    y[indices].reshape(-1))
                round_optimizer.zero_grad()
                loss.backward()
                round_optimizer.step()
        with torch.no_grad():
            accuracy = float((model.outputs(x, mask).argmax(-1) == y).float().mean())
        averages = np.mean(np.asarray(rollout_results, dtype=np.float64), axis=0)
        rounds.append({"round": round_number,
                       "scenario_seed_start": scenario_seeds.start,
                       "scenario_seed_end": scenario_seeds.stop - 1,
                       "visited_episodes": DAGGER_VISITED_EPISODES_PER_ROUND,
                       "visited_joint_states": len(visited_x),
                       "nn_teacher_action_agreement": float(np.mean(action_agreement)),
                       "aggregate_label_accuracy": accuracy,
                       "rollout_mean_deliveries": float(averages[0]),
                       "rollout_mean_collisions": float(averages[1]),
                       "rollout_shutdown_episode_rate": float(averages[2])})
    return {"source": "offline_teacher_plus_nn_visited_relabeling",
            "episodes": episodes, "epochs": epochs,
            "rows_per_role": len(labels) // 2,
            "original_teacher_label_accuracy": initial_teacher_accuracy,
            "label_accuracy": rounds[-1]["aggregate_label_accuracy"],
            "offline_teacher_recomputed_from_pre_action_state": True,
            "offline_nn_visit_actions_from_actor_only": True,
            "dagger_original_teacher_episodes": DAGGER_ORIGINAL_TEACHER_EPISODES,
            "dagger_epochs_per_round": DAGGER_EPOCHS_PER_ROUND,
            "dagger_rounds": rounds,
            "initial_role_sha256": initial,
            "pretrained_role_sha256": _weight_hashes(model)}


def collect_episode(model: DualNN, *, seed: int, generator: torch.Generator,
                    max_steps: int):
    # Train against the environment's collision and shutdown outcomes.  The
    # formal evaluator still uses unchanged participant scoring.  Neither
    # reward coefficient changes the Actors' sampled action or adds a rule.
    env = WarehouseMultiAgentEnv(participant_study_config(
        **TRAINING_REWARD_OVERRIDES))
    env.reset(seed=seed)
    marked = env.get_state()
    marked.participant_controlled_agent_id = "robot_1"
    env.set_state(marked)
    records = []
    while not (env.state.terminated or env.state.truncated) and len(records) < max_steps:
        obs = observations(env.state)
        mask = action_masks(env.state)
        with torch.no_grad():
            t_obs, t_mask = torch.from_numpy(obs)[None], torch.from_numpy(mask)[None]
            logits = model.outputs(t_obs, t_mask)[0]
            probability = torch.softmax(logits, dim=-1)
            actions_index = torch.multinomial(probability, 1, generator=generator).squeeze(-1)
            logp = torch.log(probability.gather(-1, actions_index[:, None]).squeeze(-1).clamp_min(1e-8))
            value = model.critic(t_obs.flatten(start_dim=1))[0].numpy().copy()
        actions = {agent_id: ACTIONS[int(actions_index[row])] for row, agent_id in enumerate(AGENTS)}
        _, rewards, _, _, info = env.step(actions)
        if info["proposed_actions"] != actions or info["action_override_agents"]:
            raise RuntimeError("Warehouse formal self-play action changed before physics")
        records.append({"obs": obs, "mask": mask, "action": actions_index.numpy(),
                        "logp": logp.numpy(), "value": value,
                        "reward": np.asarray([rewards[agent_id] for agent_id in AGENTS],
                                             dtype=np.float32),
                        "seed": seed, "frame": env.state.frame - 1})
    terminated = bool(env.state.terminated or env.state.truncated)
    if terminated:
        bootstrap = np.zeros(len(AGENTS), dtype=np.float32)
    else:
        with torch.no_grad():
            bootstrap = model.critic(
                torch.from_numpy(observations(env.state)).flatten()[None]
            )[0].numpy().copy()
    summary = {"seed": seed, "joint_steps": len(records), "actor_decisions": 2 * len(records),
               "proxy_human_overrides": 0, "post_policy_interventions": 0,
               "deliveries": env.state.total_deliveries,
               "collisions": env.state.robot_collision_events,
               "shutdown_episode": bool(env.state.shutdown_count),
               "team_reward": sum(float(np.mean(record["reward"])) for record in records)}
    return {"records": records, "bootstrap": bootstrap}, summary


def _sample_state(record: dict, row: int) -> dict:
    return {"obs": record["obs"][row].copy(), "mask": record["mask"][row].copy(),
            "episode": str(record["seed"])}


def _features(sample: dict) -> dict[str, float]:
    data = {name: float(value) for name, value in zip(FEATURE_NAMES, sample["obs"], strict=True)}
    data.update({f"legal_{action.lower()}": float(sample["mask"][index])
                 for index, action in enumerate(ACTIONS)})
    return data


def _oracle(model: DualNN, agent_id: str):
    def predict(sample):
        with torch.no_grad():
            raw = model.actors[agent_id](torch.from_numpy(sample["obs"]))
            mask = torch.from_numpy(sample["mask"])
            probs = torch.softmax(raw.masked_fill(mask <= 0.5, -1e9), -1).numpy()
        return {action: float(probs[index]) for index, action in enumerate(ACTIONS)}
    return predict


def _extract_selective_program(model: DualNN, agent_id: str,
                               rows: list[dict], *, pair_seed: int,
                               joint_steps: int) -> tuple[SelectiveLeafProgram | None, dict]:
    """Fit one bounded role program using only disjoint training episodes."""
    episode_ids = sorted({str(row["episode"]) for row in rows})
    if len(episode_ids) < PROGRAM_MIN_EPISODES:
        return None, {"joint_steps": joint_steps, "role": agent_id,
                      "error": f"only {len(episode_ids)} training episodes"}
    # Use the most recent 124 distinct training scenarios, grouped by seed so
    # resets within a matched 120-step block cannot cross a split.  Structure,
    # calibration, leaf eligibility and validation are mutually disjoint.
    episode_ids = episode_ids[-PROGRAM_MIN_EPISODES:]
    count = len(episode_ids)
    fit_count, calibration_count, eligibility_count, validation_count = PROGRAM_SPLIT_EPISODES
    if sum(PROGRAM_SPLIT_EPISODES) != PROGRAM_MIN_EPISODES:
        raise RuntimeError("Warehouse selective split contract changed")
    first = fit_count
    second = first + calibration_count
    third = second + eligibility_count
    groups = (episode_ids[:first], episode_ids[first:second],
              episode_ids[second:third], episode_ids[third:third + validation_count])
    membership = {episode: part for part, group in enumerate(groups)
                  for episode in group}
    parts = [[] for _ in range(4)]
    with torch.no_grad():
        for block in (rows[offset:offset + 512] for offset in range(0, len(rows), 512)):
            features = torch.from_numpy(np.stack([row["obs"] for row in block]))
            masks = torch.from_numpy(np.stack([row["mask"] for row in block]))
            probabilities = torch.softmax(
                model.actors[agent_id](features).masked_fill(masks <= .5, -1e9),
                dim=-1).numpy()
            for row, probability in zip(block, probabilities, strict=True):
                # The oracle distribution supervises the program. The
                # program's input encoder reads only `obs` and `mask`.
                part = membership.get(str(row["episode"]))
                if part is not None:
                    parts[part].append({**row, "oracle_probabilities": probability})
    try:
        program = fit_selective_program(
            agent_id, *parts, lambda row: row["oracle_probabilities"],
            actor_role_sha256=_weight_hashes(model)[agent_id],
            hard_weight=PROGRAM_HARD_WEIGHT[agent_id],
            random_seed=pair_seed + AGENTS.index(agent_id))
    except Exception as exc:
        return None, {"joint_steps": joint_steps, "role": agent_id,
                      "training_episode_count": count,
                      "error": f"{type(exc).__name__}: {exc}"}
    return program, {"joint_steps": joint_steps, "role": agent_id,
                     "training_episode_count": count,
                     "training_split_episode_counts": dict(zip(
                         ("fit", "calibration", "eligibility", "validation"),
                         PROGRAM_SPLIT_EPISODES, strict=True)),
                     "program_sha256": program.sha256(),
                     "validation": program.report["validation"],
                     "split_episode_sha256": program.report["split_episode_sha256"]}


def _program_coefficient(progress: float) -> float:
    return PROGRAM_LAMBDA * min(1.0, max(0.0,
        (progress - PROGRAM_START_FRACTION) / PROGRAM_RAMP_FRACTION))


def _program_loss(logits: torch.Tensor, mask: torch.Tensor, obs: torch.Tensor,
                  sampled_action: torch.Tensor, role: SelectiveLeafProgram | None,
                  progress: float):
    coefficient = _program_coefficient(progress)
    if role is None or coefficient <= 0.0:
        return logits.new_zeros(()), 0, 0.0
    samples = [{"obs": obs[index].detach().numpy(), "mask": mask[index].detach().numpy()}
               for index in range(len(obs))]
    actor_probs = torch.softmax(logits.detach(), -1).numpy()
    sampled = sampled_action.detach().numpy()
    targets = actor_probs.copy()
    weights = np.zeros(len(samples), dtype=np.float32)
    for index, sample in enumerate(samples):
        actor_index = int(actor_probs[index].argmax())
        if actor_index != int(sampled[index]):
            continue
        # covered() verifies the exact training-validated leaf identity,
        # physical legality, and program/current-NN action equality.
        evidence = role.covered(sample, actor_action=ACTIONS[actor_index],
                                purpose="feedback")
        if evidence is None:
            continue
        distribution = np.asarray([evidence["probabilities"][action]
                                   for action in ACTIONS], dtype=np.float32)
        distribution *= sample["mask"]
        total = float(distribution.sum())
        if total <= 1e-8:
            continue
        distribution /= total
        if int(distribution.argmax()) != actor_index:
            continue
        targets[index] = distribution
        weights[index] = 1.0
    active = int(np.count_nonzero(weights))
    if not active:
        return logits.new_zeros(()), 0, 0.0
    kl = RCPD.regularization_loss(logits, targets, weights)
    return coefficient * kl, active, float(kl.detach())


def ppo_update(model: DualNN, optimizer, episodes: list[dict],
               programs: dict[str, SelectiveLeafProgram | None] | None,
               *, progress: float, epochs: int = 4):
    rows = []
    for episode in episodes:
        advantage = np.zeros(len(AGENTS), dtype=np.float32)
        next_value = episode["bootstrap"]
        enriched = []
        for record in reversed(episode["records"]):
            delta = record["reward"] + 0.99 * next_value - record["value"]
            advantage = delta + 0.99 * 0.95 * advantage
            enriched.append((record, advantage, advantage + record["value"]))
            next_value = record["value"]
        rows.extend(reversed(enriched))
    if not rows:
        raise ValueError("PPO update needs at least one joint transition")
    obs = torch.from_numpy(np.stack([item[0]["obs"] for item in rows]))
    mask = torch.from_numpy(np.stack([item[0]["mask"] for item in rows]))
    actions = torch.as_tensor(np.stack([item[0]["action"] for item in rows]), dtype=torch.long)
    old_logp = torch.from_numpy(np.stack([item[0]["logp"] for item in rows]))
    advantages = torch.as_tensor(np.stack([item[1] for item in rows]), dtype=torch.float32)
    targets = torch.as_tensor(np.stack([item[2] for item in rows]), dtype=torch.float32)
    advantages = ((advantages - advantages.mean(dim=0, keepdim=True)) /
                  advantages.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6))
    feedback_rows = {agent_id: 0 for agent_id in AGENTS}
    feedback_kl = {agent_id: 0.0 for agent_id in AGENTS}
    feedback_window_actor_rows = {agent_id: 0 for agent_id in AGENTS}
    zero_action_coverage_rows = {agent_id: 0 for agent_id in AGENTS}
    physical_no_legal_action_rows = {agent_id: 0 for agent_id in AGENTS}
    loss_total = 0.0
    for _ in range(epochs):
        logits = model.outputs(obs, mask)
        policy = Categorical(logits=logits)
        ratio = torch.exp(policy.log_prob(actions) - old_logp)
        actor_loss = -torch.minimum(ratio * advantages,
                                    ratio.clamp(0.8, 1.2) * advantages).mean()
        critic_loss = nn.functional.mse_loss(model.critic(obs.flatten(start_dim=1)), targets)
        loss = actor_loss + 0.5 * critic_loss - 0.005 * policy.entropy().mean()
        if programs:
            for index, agent_id in enumerate(AGENTS):
                penalty, count, kl = _program_loss(logits[:, index], mask[:, index],
                                                   obs[:, index], actions[:, index],
                                                   programs[agent_id], progress)
                loss = loss + penalty
                feedback_rows[agent_id] += count
                feedback_kl[agent_id] += kl
                if _program_coefficient(progress) > 0.0:
                    # Counts are PPO role-row presentations: each rollout
                    # row is presented once per optimization epoch.  A zero
                    # means no audited, physically legal program action
                    # agreed with both masked NN argmax and executed sample.
                    exposure = len(obs)
                    feedback_window_actor_rows[agent_id] += exposure
                    zero_action_coverage_rows[agent_id] += exposure - count
                    physical_no_legal_action_rows[agent_id] += int(
                        (mask[:, index].sum(dim=-1) <= 0.5).sum())
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        loss_total += float(loss.detach())
    return {"loss": loss_total / epochs, "feedback_rows": feedback_rows,
            "feedback_kl": feedback_kl,
            "feedback_window_actor_rows": feedback_window_actor_rows,
            "zero_action_coverage_rows": zero_action_coverage_rows,
            "physical_no_legal_action_rows": physical_no_legal_action_rows}


def evaluate(model: DualNN, seeds=DEVELOPMENT_SEEDS) -> dict:
    actor = _frozen(model)
    rows = []
    for seed in seeds:
        env = WarehouseMultiAgentEnv(participant_study_config())
        env.reset(seed=int(seed))
        marked = env.get_state()
        marked.participant_controlled_agent_id = "robot_1"
        env.set_state(marked)
        while not (env.state.terminated or env.state.truncated):
            actions, _ = actor.act(env.state, base_seed=int(seed))
            _, _, _, _, info = env.step(actions)
            if info["proposed_actions"] != actions or info["action_override_agents"]:
                raise RuntimeError("Warehouse evaluator rewrote a neural action")
        rows.append({"seed": int(seed), "deliveries": env.state.total_deliveries,
                     "collisions": env.state.robot_collision_events,
                     "shutdown_episode": bool(env.state.shutdown_count)})
    summary = {"mean_deliveries": float(np.mean([row["deliveries"] for row in rows])),
               "mean_collisions": float(np.mean([row["collisions"] for row in rows])),
               "shutdown_episode_rate": float(np.mean([row["shutdown_episode"] for row in rows]))}
    summary["quality"] = (summary["mean_deliveries"] - 0.25 * summary["mean_collisions"]
                          - 5.0 * summary["shutdown_episode_rate"])
    summary["passed"] = bool(summary["mean_deliveries"] >= 5.0 and
                             summary["mean_collisions"] <= 8.0 and
                             summary["shutdown_episode_rate"] <= 0.10)
    return {"summary": summary, "episodes": rows}


def evaluate_heldout(model: DualNN) -> dict:
    """Fresh final test: 100 enrollment seeds per task with formal seed mapping."""
    tasks = {str(task): evaluate(model, seeds) for task, seeds in HELDOUT_SEEDS_BY_TASK.items()}
    rows = [{"task": int(task), "seed": base_seed,
             "scenario_seed": episode["seed"],
             **{key: value for key, value in episode.items() if key != "seed"}}
            for task, result in tasks.items()
            for base_seed, episode in zip(HELDOUT_BASE_SEEDS, result["episodes"], strict=True)]
    summary = {"mean_deliveries": float(np.mean([row["deliveries"] for row in rows])),
               "mean_collisions": float(np.mean([row["collisions"] for row in rows])),
               "shutdown_episode_rate": float(np.mean([row["shutdown_episode"] for row in rows]))}
    summary["quality"] = (summary["mean_deliveries"] - 0.25 * summary["mean_collisions"]
                          - 5.0 * summary["shutdown_episode_rate"])
    summary["passed"] = bool(summary["mean_deliveries"] >= 5.0 and
                             summary["mean_collisions"] <= 8.0 and
                             summary["shutdown_episode_rate"] <= 0.10)
    return {"summary": summary, "tasks": tasks, "episodes": rows}


def audit_selected_programs(model: DualNN, programs: dict[str, SelectiveLeafProgram],
                            *, pair_seed: int,
                            seeds=DEVELOPMENT_SEEDS) -> dict[str, dict]:
    """Audit exact frozen Actors on states separate from PPO training."""
    seeds = tuple(seeds)
    if set(programs) != set(AGENTS) or any(programs[role] is None for role in AGENTS):
        return {}
    actor_hashes = _weight_hashes(model)
    frozen = _frozen(model)
    samples = {role: [] for role in AGENTS}
    for seed in seeds:
        env = WarehouseMultiAgentEnv(participant_study_config())
        env.reset(seed=int(seed))
        marked = env.get_state()
        marked.participant_controlled_agent_id = "robot_1"
        env.set_state(marked)
        while not (env.state.terminated or env.state.truncated):
            obs, masks = observations(env.state), action_masks(env.state)
            for row, role in enumerate(AGENTS):
                samples[role].append({"obs": obs[row].copy(), "mask": masks[row].copy(),
                                      "episode": str(seed)})
            actions, _ = frozen.act(env.state, base_seed=int(seed))
            _, _, _, _, info = env.step(actions)
            if info["proposed_actions"] != actions or info["action_override_agents"]:
                raise RuntimeError("Warehouse program audit rewrote a neural action")
    audited = {}
    for row, role in enumerate(AGENTS):
        program = programs[role]
        if (program.role != role or
                program.actor_role_sha256 != actor_hashes[role] or
                program.report["max_depth"] > 5 or
                program.report["leaf_count"] > 16):
            raise RuntimeError("Warehouse selected program role, Actor, or complexity mismatch")
        measured = audit_frozen_program(program, samples[role], _oracle(model, role))
        measured["audit_source"] = ("final_independent_development_rollouts"
            if seeds == FINAL_PROGRAM_AUDIT_SEEDS else "preliminary_development_rollouts")
        measured["development_seeds"] = list(seeds)
        if measured["sample_count"] < 1000:
            measured["feedback_eligible"] = False
            measured["explanation_eligible"] = False
        audited[role] = measured
    return audited


def train_variant(initial: dict, *, pair_seed: int, variant: str,
                  joint_steps: int, development_seeds=DEVELOPMENT_SEEDS,
                  checkpoint_path: Path | None = None):
    if variant not in ("raw", "regularized"):
        raise ValueError("unknown Warehouse training variant")
    model = DualNN(hidden=int(initial["actors.robot_1.0.bias"].numel()))
    model.load_state_dict(initial)
    initial_hashes = _weight_hashes(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    generator = torch.Generator().manual_seed(pair_seed + 1)
    programs = ({agent_id: None for agent_id in AGENTS}
                if variant == "regularized" else None)
    buffers = {agent_id: deque(maxlen=PROGRAM_BUFFER_STATES) for agent_id in AGENTS}
    program_extractions = {agent_id: [] for agent_id in AGENTS}
    history, updates, development = [], [], []
    cumulative_feedback = {agent_id: 0 for agent_id in AGENTS}
    best = None
    used_steps = 0
    episode_number = 0
    next_eval = max(1, joint_steps // 10)
    if checkpoint_path is not None and checkpoint_path.is_file():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (saved.get("pair_seed") != pair_seed or saved.get("variant") != variant or
            saved.get("joint_steps") != joint_steps or
            saved.get("scenario_schedule_version") != SCENARIO_SCHEDULE_VERSION or
            saved.get("training_reward_version") != TRAINING_REWARD_VERSION or
            saved.get("bc_architecture_version") != BC_ARCHITECTURE_VERSION or
            (variant == "regularized" and
             saved.get("program_feedback_version") != PROGRAM_FEEDBACK_VERSION) or
            saved.get("initial_role_sha256") != initial_hashes):
            raise RuntimeError("Warehouse resume checkpoint belongs to another experiment")
        if saved.get("complete"):
            model.load_state_dict(saved["selected_state_dict"])
            selected_hashes = _weight_hashes(model)
            saved_programs = {agent_id: SelectiveLeafProgram.from_dict(
                                  payload, actor_role_sha256=selected_hashes[agent_id])
                              for agent_id, payload in saved.get("selected_programs", {}).items()}
            report = saved["report"]
            if variant == "raw":
                # Earlier raw-only checkpoints predate these accounting
                # fields. Raw never enters a program-feedback window.
                for field in ("feedback_window_actor_rows",
                              "zero_action_coverage_rows",
                              "physical_no_legal_action_rows"):
                    report.setdefault(field, {role: 0 for role in AGENTS})
                report.setdefault("zero_action_coverage_definition",
                                  ZERO_ACTION_COVERAGE_DEFINITION)
            norms, changed = _actor_weight_changes(initial, model)
            report.setdefault("actor_weight_delta_l2", norms)
            report.setdefault("actor_weight_changed_parameter_count", changed)
            return model, saved_programs, report
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["generator_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        used_steps = int(saved["used_steps"])
        episode_number = int(saved["episode_number"])
        history, updates, development = saved["history"], saved["updates"], saved["development"]
        cumulative_feedback = {role: sum(item["feedback_rows"][role] for item in updates)
                               for role in AGENTS}
        best = saved["best"]
        if best is not None and "selection_rank" not in best:
            previous = best["development"]
            active = bool(best["feedback_active"])
            best["selection_rank"] = (
                bool(previous["passed"] and active), bool(previous["passed"]),
                active, float(previous["quality"]))
        next_eval = int(saved["next_eval"])
        if programs is not None:
            for agent_id in AGENTS:
                payload = saved["programs"].get(agent_id)
                programs[agent_id] = (SelectiveLeafProgram.from_dict(
                    payload, actor_role_sha256=payload["actor_role_sha256"])
                    if payload is not None else None)
                buffers[agent_id].extend(saved["buffers"][agent_id])
            program_extractions = saved["program_extractions"]
        print(json.dumps({"domain": "warehouse", "variant": variant, "pair_seed": pair_seed,
                          "resumed_joint_steps": used_steps}), flush=True)
    while used_steps < joint_steps:
        batch = []
        while used_steps < joint_steps and len(batch) < 16:
            seed = _scenario_seed(pair_seed, used_steps)
            block_remaining = SCENARIO_BLOCK_STEPS - used_steps % SCENARIO_BLOCK_STEPS
            episode, summary = collect_episode(model, seed=seed, generator=generator,
                                               max_steps=min(block_remaining, joint_steps - used_steps))
            if not summary["joint_steps"]:
                raise RuntimeError("Warehouse rollout yielded no steps")
            batch.append(episode)
            history.append(summary)
            used_steps += summary["joint_steps"]
            episode_number += 1
            if programs is not None:
                for record in episode["records"]:
                    for row, agent_id in enumerate(AGENTS):
                        buffers[agent_id].append(_sample_state(record, row))
        update = ppo_update(model, optimizer, batch, programs,
                            progress=used_steps / joint_steps)
        update["joint_steps"] = used_steps
        updates.append(update)
        for agent_id in AGENTS:
            cumulative_feedback[agent_id] += update["feedback_rows"][agent_id]
        if used_steps >= next_eval or used_steps == joint_steps:
            if programs is not None:
                # Extract after the PPO update so the program bound into a
                # selected checkpoint audits these exact frozen Actor hashes.
                for agent_id in AGENTS:
                    programs[agent_id], extracted = _extract_selective_program(
                        model, agent_id, list(buffers[agent_id]),
                        pair_seed=pair_seed, joint_steps=used_steps)
                    program_extractions[agent_id].append(extracted)
            measured = evaluate(model, development_seeds)
            result = {"joint_steps": used_steps, **measured["summary"]}
            development.append(result)
            # Raw may use any PPO checkpoint chosen on development data.
            # Regularized must select after the feedback window. A passing
            # model with real feedback outranks all other checkpoints; among
            # other candidates, a passing model outranks higher scalar
            # quality from a checkpoint that violates a hard safety gate.
            feedback_active = variant == "raw" or all(cumulative_feedback.values())
            selectable = variant == "raw" or used_steps / joint_steps >= 0.85
            rank = (bool(result["passed"] and feedback_active),
                    bool(result["passed"]), bool(feedback_active),
                    float(result["quality"]))
            if (selectable and
                (best is None or rank > best["selection_rank"])):
                best = {"state_dict": deepcopy(model.state_dict()),
                        "development": result,
                        "selection_rank": rank,
                        "feedback_active": feedback_active,
                        "selected_feedback_rows": cumulative_feedback.copy(),
                        "programs": ({role: deepcopy(program)
                                      for role, program in programs.items()}
                                     if programs is not None else {}),
                        "program_audit": ({role: (program.report if program else None)
                                           for role, program in programs.items()}
                                          if programs is not None else {})}
            next_eval += max(1, joint_steps // 10)
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {"pair_seed": pair_seed, "variant": variant, "joint_steps": joint_steps,
                           "scenario_schedule_version": SCENARIO_SCHEDULE_VERSION,
                           "training_reward_version": TRAINING_REWARD_VERSION,
                           "bc_architecture_version": BC_ARCHITECTURE_VERSION,
                           "program_feedback_version": PROGRAM_FEEDBACK_VERSION,
                           "initial_role_sha256": initial_hashes, "complete": False,
                           "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "generator_state": generator.get_state(),
                           "torch_rng_state": torch.get_rng_state(),
                           "used_steps": used_steps, "episode_number": episode_number,
                           "history": history, "updates": updates, "development": development,
                           "best": best, "next_eval": next_eval,
                           "programs": ({role: (programs[role].to_dict()
                                                  if programs[role] is not None else None)
                                         for role in AGENTS}
                                        if programs is not None else {}),
                           "program_extractions": program_extractions,
                           "buffers": ({role: list(buffers[role]) for role in AGENTS}
                                       if programs is not None else {})}
                temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
                torch.save(payload, temporary)
                temporary.replace(checkpoint_path)
            print(json.dumps({"domain": "warehouse", "variant": variant, "pair_seed": pair_seed,
                              "joint_steps": used_steps, "development": result,
                              "feedback_rows": update["feedback_rows"]}), flush=True)
    if best is None:
        raise RuntimeError("Warehouse trainer did not produce a late development checkpoint")
    _validate_scenario_schedule(history, pair_seed, joint_steps)
    model.load_state_dict(best["state_dict"])
    weight_delta_l2, weight_changed_count = _actor_weight_changes(initial, model)
    report = {"domain": "warehouse", "variant": variant, "pair_seed": pair_seed,
              "joint_steps": used_steps, "actor_decisions": 2 * used_steps,
              "scenario_schedule_version": SCENARIO_SCHEDULE_VERSION,
              "training_reward_version": TRAINING_REWARD_VERSION,
              "training_reward_overrides": TRAINING_REWARD_OVERRIDES.copy(),
              "bc_architecture_version": BC_ARCHITECTURE_VERSION,
              "program_feedback_version": PROGRAM_FEEDBACK_VERSION,
              "scenario_schedule_sha256": _scenario_schedule_sha256(pair_seed, joint_steps),
              "initial_role_sha256": initial_hashes, "selected_role_sha256": _weight_hashes(model),
              "actor_weight_delta_l2": weight_delta_l2,
              "actor_weight_changed_parameter_count": weight_changed_count,
              "development": development, "selected_development": best["development"],
              "extraction_program_audit": best.get("program_audit", {}),
              "training_episodes": history, "updates": updates,
              "proxy_human_overrides": 0, "post_policy_interventions": 0,
              "feedback_rows": {role: sum(item["feedback_rows"][role] for item in updates)
                                for role in AGENTS},
              "selected_feedback_rows": best["selected_feedback_rows"],
              "feedback_window_actor_rows": {
                  role: sum(item.get("feedback_window_actor_rows", {}).get(role, 0)
                            for item in updates) for role in AGENTS},
              "zero_action_coverage_rows": {
                  role: sum(item.get("zero_action_coverage_rows", {}).get(role, 0)
                            for item in updates) for role in AGENTS},
              "physical_no_legal_action_rows": {
                  role: sum(item.get("physical_no_legal_action_rows", {}).get(role, 0)
                            for item in updates) for role in AGENTS},
              "zero_action_coverage_definition": ZERO_ACTION_COVERAGE_DEFINITION,
              "program_extractions": program_extractions,
              "rcpd": ({role: (best["programs"][role].report
                                if best["programs"][role] else None)
                        for role in AGENTS}
                       if programs is not None else None)}
    if variant == "raw" and any(report["feedback_rows"].values()):
        raise RuntimeError("raw Warehouse baseline received program feedback")
    if checkpoint_path is not None:
        complete = {"pair_seed": pair_seed, "variant": variant, "joint_steps": joint_steps,
                    "scenario_schedule_version": SCENARIO_SCHEDULE_VERSION,
                    "training_reward_version": TRAINING_REWARD_VERSION,
                    "bc_architecture_version": BC_ARCHITECTURE_VERSION,
                    "program_feedback_version": PROGRAM_FEEDBACK_VERSION,
                    "initial_role_sha256": initial_hashes, "complete": True,
                    "selected_state_dict": model.state_dict(),
                    "selected_programs": {role: program.to_dict()
                                          for role, program in best["programs"].items()
                                          if program is not None},
                    "report": report}
        temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        torch.save(complete, temporary)
        temporary.replace(checkpoint_path)
    return model, best["programs"], report


def export_actor(model: DualNN, path: Path, *, pair_seed: int, variant: str) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = {agent_id: _role_weights(model, agent_id) for agent_id in AGENTS}
    role_hashes = {agent_id: role_weight_sha256(weight) for agent_id, weight in weights.items()}
    metadata = {"model_version": DUAL_MODEL_VERSION,
                "observation_version": OBSERVATION_VERSION,
                "observation_dim": OBSERVATION_DIM,
                "map_layout_id": STUDY_MAP_LAYOUT.layout_id,
                "action_names": list(ACTIONS), "agent_ids": list(AGENTS),
                "role_sha256": role_hashes, "pair_seed": pair_seed, "variant": variant}
    payload = {f"{agent_id}_{key}": value for agent_id, group in weights.items()
               for key, value in group.items()}
    np.savez_compressed(path, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)), **payload)
    digest = sha256(path.read_bytes()).hexdigest()
    frozen = FrozenDualPureNN.load(path, expected_sha256=digest,
                                   expected_role_sha256=role_hashes)
    probe = np.random.default_rng(17).normal(size=(2, OBSERVATION_DIM)).astype(np.float32)
    with torch.no_grad():
        expected = np.stack([model.actors[agent_id](torch.from_numpy(probe[index])).numpy()
                             for index, agent_id in enumerate(AGENTS)])
    if not np.allclose(frozen.logits(probe), expected, rtol=1e-5, atol=1e-5):
        raise RuntimeError("Warehouse v5 PyTorch/NumPy logit parity failed")
    return {"path": str(path), "sha256": digest, "role_sha256": role_hashes,
            "maximum_logit_difference": float(np.max(np.abs(frozen.logits(probe) - expected)))}


def _write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return sha256(path.read_bytes()).hexdigest()


def _heldout_selection(selected: dict) -> dict:
    return {
        "release": "policylens-three-domain-20261014.v5.0",
        "selected_pair_seed": selected["seed"],
        "selected_role_sha256": {
            variant: selected[variant]["report"]["selected_role_sha256"]
            for variant in ("raw", "regularized")
        },
        "scenario_schedule_sha256": {
            variant: selected[variant]["report"]["scenario_schedule_sha256"]
            for variant in ("raw", "regularized")
        },
        "selected_program_sha256": {
            role: sha256(json.dumps(
                selected["regularized"]["programs"][role].to_dict(),
                sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            for role in AGENTS
        },
        "heldout_scenario_seed_rule": "enrollment_seed_times_10_plus_task",
        "heldout_seed_ranges": {
            str(task): [HELDOUT_BASE_SEEDS[0], HELDOUT_BASE_SEEDS[-1]]
            for task in HELDOUT_SEEDS_BY_TASK
        },
    }


def _selection_from_acceptance(acceptance: dict) -> dict:
    """Reconstruct the entire once-only receipt binding from saved evidence."""
    try:
        selected = [item for item in acceptance["candidate_training"]
                    if item["pair_seed"] == acceptance["selected_pair_seed"]]
        if len(selected) != 1:
            raise ValueError("selected pair is not unique")
        selection = {
            "release": acceptance["release"],
            "selected_pair_seed": acceptance["selected_pair_seed"],
            "selected_role_sha256": acceptance["role_sha256"],
            "scenario_schedule_sha256": {
                variant: selected[0]["variants"][variant]["scenario_schedule_sha256"]
                for variant in ("raw", "regularized")},
            "selected_program_sha256": acceptance["program_sha256"]["regularized"],
            "heldout_scenario_seed_rule": acceptance["heldout_scenario_seed_rule"],
            "heldout_seed_ranges": acceptance["heldout_seed_ranges"],
        }
        if (selection["release"] != "policylens-three-domain-20261014.v5.0"
                or selection["heldout_scenario_seed_rule"] !=
                    "enrollment_seed_times_10_plus_task"
                or selection["heldout_seed_ranges"] != {
                    str(task): [HELDOUT_BASE_SEEDS[0], HELDOUT_BASE_SEEDS[-1]]
                    for task in HELDOUT_SEEDS_BY_TASK}
                or set(selection["selected_role_sha256"]) != {"raw", "regularized"}
                or any(selection["selected_role_sha256"][variant] !=
                       selected[0]["variants"][variant]["selected_role_sha256"]
                       for variant in ("raw", "regularized"))
                or set(selection["selected_program_sha256"]) != set(AGENTS)):
            raise ValueError("accepted assets or held-out seed mapping changed")
        return selection
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Warehouse acceptance cannot reconstruct once-only selection") from exc


def _install_accepted_assets(acceptance: dict, output_dir: Path) -> None:
    """Install an accepted frozen pair, including after a safe completed-run resume."""
    if (not acceptance.get("passed") or
            output_dir.resolve() != (ROOT / "output/deployment").resolve()):
        raise RuntimeError("Only an accepted full-budget deployment run may install v5 policy hashes")
    _selection_from_acceptance(acceptance)
    for variant in ("raw", "regularized"):
        actor_path = output_dir / f"warehouse_dual_v5_{variant}_actor.npz"
        expected = acceptance["artifact_sha256"][variant]
        if sha256(actor_path.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Warehouse accepted Actor file changed before install")
        FrozenDualPureNN.load(actor_path, expected_sha256=expected,
                              expected_role_sha256=acceptance["role_sha256"][variant])
        report_path = output_dir / f"warehouse_dual_v5_{variant}_training.json"
        if sha256(report_path.read_bytes()).hexdigest() != acceptance[
                "training_report_sha256"][variant]:
            raise RuntimeError("Warehouse accepted training report changed before install")
    for role, digest in acceptance["program_sha256"]["regularized"].items():
        path = output_dir / f"warehouse_dual_v5_regularized_{role}_program.json"
        if sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Warehouse accepted program changed before install")
    config = json.loads(V5_CONFIG.read_text())
    for variant in ("raw", "regularized"):
        config["variants"][variant]["actor_sha256"] = acceptance[
            "artifact_sha256"][variant]
        config["variants"][variant]["role_sha256"] = acceptance[
            "role_sha256"][variant]
    _write_json(V5_CONFIG, config)


def _final_program_audit_selection(selected: dict) -> dict:
    selection = _heldout_selection(selected)
    selection["final_program_audit_seeds"] = list(FINAL_PROGRAM_AUDIT_SEEDS)
    return selection


def _start_final_program_audit_once(path: Path, selection: dict) -> None:
    """Claim the untouched program-validation block before its first rollout."""
    payload = {"status": "started", "selection": selection}
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("Warehouse final program audit receipt already exists") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_final_program_audit(audit: dict, selection: dict) -> None:
    """Check frozen program evidence against the exact claimed release assets."""
    if (set(audit) != set(AGENTS) or
            selection.get("final_program_audit_seeds") != list(FINAL_PROGRAM_AUDIT_SEEDS)):
        raise RuntimeError("Warehouse final program audit selection is incomplete")
    for role in AGENTS:
        row = audit[role]
        validation = row.get("validation", {})
        if (row.get("audit_source") != "final_independent_development_rollouts"
                or row.get("role") != role
                or row.get("development_seeds") != list(FINAL_PROGRAM_AUDIT_SEEDS)
                or row.get("actor_role_sha256") !=
                    selection["selected_role_sha256"]["regularized"][role]
                or row.get("program_sha256") !=
                    selection["selected_program_sha256"][role]
                or not isinstance(row.get("sample_count"), int)
                or row["sample_count"] < 1000
                or not 0 <= row.get("program_depth", 6) <= 5
                or not 1 <= row.get("program_leaf_count", 17) <= 16
                or validation.get("global", {}).get("rows") != row["sample_count"]
                or row.get("global") != validation.get("global")
                or row.get("feedback") != validation.get("feedback")
                or row.get("explanation") != validation.get("explanation")
                or row.get("feedback_eligible") != validation.get("feedback_passed")
                or row.get("explanation_eligible") != validation.get("explanation_passed")
                or row.get("passed") != bool(row["feedback_eligible"] and
                       (role != "robot_2" or row["explanation_eligible"]))):
            raise RuntimeError("Warehouse final program audit asset or evidence mismatch")
        for purpose, agreement in (("feedback", .85), ("explanation", .90)):
            if purpose == "explanation" and role != "robot_2":
                if row["explanation_eligible"]:
                    raise RuntimeError("Warehouse human-role program cannot explain AI")
                continue
            metrics = validation.get(purpose, {})
            if row[purpose + "_eligible"] and (
                    metrics.get("coverage", 0) < .20 or
                    metrics.get("agreement", 0) < agreement or
                    metrics.get("kl") is None or
                    not np.isfinite(metrics["kl"]) or metrics["kl"] > .35 or
                    metrics.get("illegal_count") != 0 or
                    metrics.get("rows", 0) <= 0):
                raise RuntimeError("Warehouse final program audit numerical gate mismatch")


def _complete_final_program_audit(path: Path, selection: dict, audit: dict) -> None:
    _validate_final_program_audit(audit, selection)
    if json.loads(path.read_text()) != {"status": "started", "selection": selection}:
        raise RuntimeError("Warehouse final program audit receipt changed during evaluation")
    temporary = path.with_suffix(path.suffix + ".complete.tmp")
    evidence_hash = sha256(json.dumps(audit, sort_keys=True).encode()).hexdigest()
    _write_json(temporary, {"status": "completed", "selection": selection,
                            "audit": audit, "audit_sha256": evidence_hash})
    temporary.replace(path)


def _reuse_final_program_audit(path: Path, selection: dict) -> dict:
    receipt = json.loads(path.read_text())
    if receipt.get("status") != "completed" or receipt.get("selection") != selection:
        raise RuntimeError("Warehouse final program audit is incomplete or selected assets changed")
    audit = receipt.get("audit")
    if (not isinstance(audit, dict) or set(audit) != set(AGENTS) or
            any(audit[role].get("development_seeds") != list(FINAL_PROGRAM_AUDIT_SEEDS)
                for role in AGENTS)):
        raise RuntimeError("Warehouse final program audit evidence is incomplete")
    if receipt.get("audit_sha256") != sha256(
            json.dumps(audit, sort_keys=True).encode()).hexdigest():
        raise RuntimeError("Warehouse final program audit evidence hash mismatch")
    _validate_final_program_audit(audit, selection)
    return audit


def _start_heldout_once(path: Path, selection: dict) -> None:
    """Claim the fresh test set atomically before the first test rollout."""
    payload = {"status": "started", "selection": selection}
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("Warehouse held-out receipt already exists; do not re-evaluate") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _validate_heldout_evidence(heldout: dict) -> None:
    """Check the fixed 3×100 task map and all reported scores in a receipt."""
    if not isinstance(heldout, dict) or set(heldout) != {"raw", "regularized"}:
        raise RuntimeError("Warehouse completed held-out evidence is incomplete")

    def summary_matches(summary: dict, rows: list[dict]) -> bool:
        if not isinstance(summary, dict) or not rows:
            return False
        expected = {
            "mean_deliveries": float(np.mean([row["deliveries"] for row in rows])),
            "mean_collisions": float(np.mean([row["collisions"] for row in rows])),
            "shutdown_episode_rate": float(np.mean([
                row["shutdown_episode"] for row in rows])),
        }
        expected["quality"] = (expected["mean_deliveries"] -
                               .25 * expected["mean_collisions"] -
                               5.0 * expected["shutdown_episode_rate"])
        expected["passed"] = bool(expected["mean_deliveries"] >= 5.0 and
                                  expected["mean_collisions"] <= 8.0 and
                                  expected["shutdown_episode_rate"] <= .10)
        return (set(summary) == set(expected) and
                summary["passed"] is expected["passed"] and
                all(isinstance(summary[key], (int, float)) and
                    np.isfinite(summary[key]) and
                    np.isclose(summary[key], expected[key], rtol=0, atol=1e-12)
                    for key in expected if key != "passed"))

    for variant in ("raw", "regularized"):
        evidence = heldout[variant]
        if (not isinstance(evidence, dict) or
                set(evidence.get("tasks", {})) != {"1", "2", "3"} or
                len(evidence.get("episodes", [])) != 300):
            raise RuntimeError("Warehouse completed held-out evidence is incomplete")
        combined = []
        for task, scenario_seeds in HELDOUT_SEEDS_BY_TASK.items():
            task_result = evidence["tasks"][str(task)]
            episodes = task_result.get("episodes", [])
            if len(episodes) != len(HELDOUT_BASE_SEEDS):
                raise RuntimeError("Warehouse held-out task seed count changed")
            for base_seed, scenario_seed, row in zip(
                    HELDOUT_BASE_SEEDS, scenario_seeds, episodes, strict=True):
                if (not isinstance(row, dict) or
                        set(row) != {"seed", "deliveries", "collisions", "shutdown_episode"} or
                        row["seed"] != scenario_seed or
                        type(row["deliveries"]) is not int or row["deliveries"] < 0 or
                        type(row["collisions"]) is not int or row["collisions"] < 0 or
                        type(row["shutdown_episode"]) is not bool):
                    raise RuntimeError("Warehouse held-out task episode mapping changed")
                combined.append({"task": task, "seed": base_seed,
                                 "scenario_seed": scenario_seed,
                                 "deliveries": row["deliveries"],
                                 "collisions": row["collisions"],
                                 "shutdown_episode": row["shutdown_episode"]})
            if not summary_matches(task_result.get("summary"), episodes):
                raise RuntimeError("Warehouse held-out task summary changed")
        if (evidence["episodes"] != combined or
                not summary_matches(evidence.get("summary"), combined)):
            raise RuntimeError("Warehouse held-out overall evidence changed")


def _complete_heldout(path: Path, selection: dict, heldout: dict) -> None:
    _validate_heldout_evidence(heldout)
    current = json.loads(path.read_text())
    if current != {"status": "started", "selection": selection}:
        raise RuntimeError("Warehouse held-out start receipt changed during evaluation")
    temporary = path.with_suffix(path.suffix + ".complete.tmp")
    evidence_hash = sha256(json.dumps(heldout, sort_keys=True).encode()).hexdigest()
    _write_json(temporary, {"status": "completed", "selection": selection,
                            "heldout": heldout, "heldout_sha256": evidence_hash})
    temporary.replace(path)


def _reuse_completed_heldout(path: Path, selection: dict) -> dict:
    receipt = json.loads(path.read_text())
    if receipt.get("status") != "completed" or receipt.get("selection") != selection:
        raise RuntimeError("Warehouse held-out receipt is incomplete or selected models changed")
    heldout = receipt.get("heldout")
    if receipt.get("heldout_sha256") != sha256(
            json.dumps(heldout, sort_keys=True).encode()).hexdigest():
        raise RuntimeError("Warehouse completed held-out evidence hash mismatch")
    _validate_heldout_evidence(heldout)
    return heldout


def _choose_quality_pair(candidates: list[dict]) -> tuple[dict, bool]:
    """Select by the weaker variant's development quality, before program audit."""
    if not candidates:
        raise ValueError("Warehouse pair selection needs candidates")
    eligible = [item for item in candidates if item["development_passed"]]
    return max(eligible or candidates, key=lambda item: item["min_quality"]), bool(eligible)


def _all_candidates_received_regularized_feedback(candidates: list[dict]) -> bool:
    """A one-time final audit requires real feedback for both roles in all pairs."""
    return bool(candidates) and all(
        all(item["regularized"]["report"]["feedback_rows"][role] > 0
            for role in AGENTS)
        for item in candidates)


def run_experiment(*, pair_seeds=DEFAULT_SEEDS, joint_steps=FULL_JOINT_STEPS,
                   bc_episodes=200, bc_epochs=30, output_dir: Path,
                   install: bool = False, enable_heldout: bool = False):
    torch.set_num_threads(1)
    if len(pair_seeds) != 3 or len(set(pair_seeds)) != 3:
        raise ValueError("Warehouse experiment requires exactly three unique paired seeds")
    if joint_steps <= 0 or bc_episodes <= 0 or bc_epochs <= 0:
        raise ValueError("training budgets must be positive")
    if enable_heldout and output_dir.resolve() != (ROOT / "output/deployment").resolve():
        raise RuntimeError("Warehouse final audit and held-out require canonical deployment directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "warehouse_dual_v5_acceptance.json"
    heldout_receipt_path = output_dir / "warehouse_dual_v5_heldout_receipt.json"
    program_audit_receipt_path = output_dir / "warehouse_dual_v5_final_program_audit_receipt.json"
    if receipt_path.is_file():
        previous_receipt = json.loads(receipt_path.read_text())
        if previous_receipt.get("heldout_evaluated"):
            if not heldout_receipt_path.is_file():
                raise RuntimeError("Warehouse held-out set was already evaluated; receipt missing")
            pinned = json.loads(heldout_receipt_path.read_text())
            full_selection = _selection_from_acceptance(previous_receipt)
            if (pinned.get("status") != "completed" or
                pinned.get("selection") != full_selection or
                pinned.get("heldout") != previous_receipt.get("heldout") or
                pinned.get("heldout_sha256") != sha256(json.dumps(
                    previous_receipt.get("heldout"), sort_keys=True).encode()).hexdigest()):
                raise RuntimeError("Warehouse held-out receipt and acceptance disagree")
            _reuse_completed_heldout(heldout_receipt_path, full_selection)
            if not program_audit_receipt_path.is_file():
                raise RuntimeError("Warehouse accepted run lacks independent program audit receipt")
            program_receipt = json.loads(program_audit_receipt_path.read_text())
            final_audit = previous_receipt.get("final_program_audit")
            if (program_receipt.get("status") != "completed" or
                program_receipt.get("selection") != (full_selection | {
                    "final_program_audit_seeds": list(FINAL_PROGRAM_AUDIT_SEEDS)}) or
                program_receipt.get("audit") != final_audit or
                program_receipt.get("audit_sha256") != sha256(json.dumps(
                    final_audit, sort_keys=True).encode()).hexdigest()):
                raise RuntimeError("Warehouse final program audit and acceptance disagree")
            _validate_final_program_audit(final_audit, program_receipt["selection"])
            if install:
                _install_accepted_assets(previous_receipt, output_dir)
            return previous_receipt
    if heldout_receipt_path.is_file():
        existing = json.loads(heldout_receipt_path.read_text())
        if existing.get("status") != "completed":
            raise RuntimeError("Warehouse held-out started receipt exists; do not re-evaluate")
    candidates = []
    for pair_seed in pair_seeds:
        torch.manual_seed(pair_seed)
        initial_model = DualNN()
        work_dir = output_dir / "_warehouse_dual_v5_work"
        work_dir.mkdir(parents=True, exist_ok=True)
        initial_path = work_dir / f"pair_{pair_seed}_bc_rolecritic_dagger_v3.pt"
        if initial_path.is_file():
            saved = torch.load(initial_path, map_location="cpu", weights_only=False)
            if (saved.get("pair_seed") != pair_seed or saved.get("bc_episodes") != bc_episodes or
                saved.get("bc_epochs") != bc_epochs or
                saved.get("architecture_version") != BC_ARCHITECTURE_VERSION):
                raise RuntimeError("Warehouse BC cache belongs to another experiment")
            initial_model.load_state_dict(saved["state_dict"])
            bc = saved["report"]
        else:
            bc = pretrain_pair(initial_model, seed=pair_seed, episodes=bc_episodes,
                               epochs=bc_epochs)
            temporary = initial_path.with_suffix(".pt.tmp")
            torch.save({"pair_seed": pair_seed, "bc_episodes": bc_episodes,
                        "architecture_version": BC_ARCHITECTURE_VERSION,
                        "bc_epochs": bc_epochs, "state_dict": initial_model.state_dict(),
                        "report": bc}, temporary)
            temporary.replace(initial_path)
        initial = deepcopy(initial_model.state_dict())
        pair = {"seed": pair_seed, "bc": bc}
        for variant in ("raw", "regularized"):
            model, programs, report = train_variant(
                initial, pair_seed=pair_seed, variant=variant, joint_steps=joint_steps,
                checkpoint_path=work_dir / f"pair_{pair_seed}_{variant}_matched.pt")
            if report["initial_role_sha256"] != bc["pretrained_role_sha256"]:
                raise RuntimeError("paired variants did not share the same Actor initialization")
            pair[variant] = {"model": model, "programs": programs, "report": report}
        pair["development_passed"] = all(pair[v]["report"]["selected_development"]["passed"]
                                         for v in ("raw", "regularized"))
        pair["selected_program_audit"] = (
            audit_selected_programs(pair["regularized"]["model"],
                                    pair["regularized"]["programs"], pair_seed=pair_seed)
            if pair["development_passed"] else {})
        if (pair["raw"]["report"]["scenario_schedule_sha256"] !=
                pair["regularized"]["report"]["scenario_schedule_sha256"]):
            raise RuntimeError("paired Warehouse variants did not share the same scenario schedule")
        pair["min_quality"] = min(pair[v]["report"]["selected_development"]["quality"]
                                  for v in ("raw", "regularized"))
        candidates.append(pair)
    # The paired seed is chosen strictly on model quality among pairs whose
    # raw and regularized Actors both pass development gates. Program failure
    # on that chosen pair blocks release; it cannot switch us to a lower
    # quality pair after seeing explanation metrics.
    selected, model_pair_passed = _choose_quality_pair(candidates)
    all_candidate_feedback_passed = _all_candidates_received_regularized_feedback(candidates)
    preliminary_program_passed = bool(
        model_pair_passed and
        all_candidate_feedback_passed and
        all(selected["regularized"]["report"]["selected_feedback_rows"].values()) and
        all((selected["selected_program_audit"].get(role) or {}).get("feedback_eligible")
            for role in AGENTS) and
        (selected["selected_program_audit"].get("robot_2") or {}).get(
            "explanation_eligible", False))
    full_budget = joint_steps == FULL_JOINT_STEPS
    # Development gates select one pair before the untouched independent
    # program audit. That audit is claimed once for this exact selected pair;
    # a failure blocks held-out without trying another pair on fresh data.
    final_program_audit = {}
    if preliminary_program_passed and full_budget and enable_heldout:
        audit_selection = _final_program_audit_selection(selected)
        if program_audit_receipt_path.is_file():
            final_program_audit = _reuse_final_program_audit(
                program_audit_receipt_path, audit_selection)
        else:
            _start_final_program_audit_once(program_audit_receipt_path,
                                            audit_selection)
            final_program_audit = audit_selected_programs(
                selected["regularized"]["model"],
                selected["regularized"]["programs"], pair_seed=selected["seed"],
                seeds=FINAL_PROGRAM_AUDIT_SEEDS)
            _complete_final_program_audit(program_audit_receipt_path,
                                          audit_selection, final_program_audit)
    final_program_passed = bool(
        all((final_program_audit.get(role) or {}).get("feedback_eligible")
            for role in AGENTS) and
        (final_program_audit.get("robot_2") or {}).get("explanation_eligible"))
    heldout_evaluated = bool(preliminary_program_passed and full_budget and enable_heldout and
                            final_program_passed)
    if heldout_receipt_path.is_file() and not heldout_evaluated:
        raise RuntimeError("Warehouse held-out receipt exists but development gates no longer pass")
    heldout = {}
    if heldout_evaluated:
        selection = _heldout_selection(selected)
        if heldout_receipt_path.is_file():
            heldout = _reuse_completed_heldout(heldout_receipt_path, selection)
        else:
            _start_heldout_once(heldout_receipt_path, selection)
            heldout = {variant: evaluate_heldout(selected[variant]["model"])
                       for variant in ("raw", "regularized")}
            _complete_heldout(heldout_receipt_path, selection, heldout)
    artifacts, reports, program_hashes = {}, {}, {}
    for variant in ("raw", "regularized"):
        run = selected[variant]
        actor_path = output_dir / f"warehouse_dual_v5_{variant}_actor.npz"
        artifacts[variant] = export_actor(run["model"], actor_path,
                                          pair_seed=selected["seed"], variant=variant)
        report_path = output_dir / f"warehouse_dual_v5_{variant}_training.json"
        report = run["report"] | {"bc": selected["bc"],
                                  "bc_teacher_observability_probe":
                                      BC_TEACHER_OBSERVABILITY_PROBE,
                                  "selected_pair_seed": selected["seed"],
                                  "artifact": artifacts[variant],
                                  "selected_program_audit": (final_program_audit
                                      if variant == "regularized" else {}),
                                  "preliminary_program_audit": (selected["selected_program_audit"]
                                      if variant == "regularized" else {}),
                                  "candidate_development": [
                                      {"pair_seed": item["seed"],
                                       "selected_development": item[variant]["report"]["selected_development"],
                                       "feedback_rows": item[variant]["report"]["feedback_rows"],
                                       "feedback_window_actor_rows": item[variant]["report"][
                                           "feedback_window_actor_rows"],
                                       "zero_action_coverage_rows": item[variant]["report"][
                                           "zero_action_coverage_rows"]}
                                      for item in candidates]}
        reports[variant] = {"path": str(report_path), "sha256": _write_json(report_path, report)}
        if variant == "regularized":
            program_hashes[variant] = {}
            for agent_id in AGENTS:
                program = run["programs"].get(agent_id)
                if program is not None:
                    path = output_dir / f"warehouse_dual_v5_{variant}_{agent_id}_program.json"
                    program_hashes[variant][agent_id] = program.save(path)
    passed = bool(heldout_evaluated and all(heldout[v]["summary"]["passed"]
                                             for v in ("raw", "regularized")) and
                  all_candidate_feedback_passed and
                  all(item[variant]["report"]["joint_steps"] == FULL_JOINT_STEPS
                      and all(item[variant]["report"]["selected_role_sha256"][role]
                              != item[variant]["report"]["initial_role_sha256"][role]
                              for role in AGENTS)
                      for item in candidates for variant in ("raw", "regularized")) and
                  len(program_hashes.get("regularized", {})) == 2 and
                  all(selected["regularized"]["report"]["selected_feedback_rows"].values()) and
                  all((final_program_audit.get(role) or {}).get(
                      "feedback_eligible", False) for role in AGENTS) and
                  (final_program_audit.get("robot_2") or {}).get(
                      "explanation_eligible", False))
    acceptance = {"release": "policylens-three-domain-20261014.v5.0",
                  "domain": "warehouse", "passed": passed,
                  "all_candidate_regularized_feedback_passed": all_candidate_feedback_passed,
                  "heldout_locked": not heldout_evaluated,
                  "heldout_evaluated": heldout_evaluated,
                  "final_program_audit": final_program_audit,
                  "heldout_seed_ranges": {str(task): [HELDOUT_BASE_SEEDS[0],
                                                        HELDOUT_BASE_SEEDS[-1]]
                                          for task in HELDOUT_SEEDS_BY_TASK},
                  "heldout_scenario_seed_rule": "enrollment_seed_times_10_plus_task",
                  "pair_seeds": list(pair_seeds), "selected_pair_seed": selected["seed"],
                  "bc_teacher_observability_probe": BC_TEACHER_OBSERVABILITY_PROBE,
                  "candidate_training": [
                      {"pair_seed": item["seed"],
                       "initial_role_sha256": item["bc"]["pretrained_role_sha256"],
                       "variants": {
                           variant: {
                               "joint_steps": item[variant]["report"]["joint_steps"],
                               "actor_decisions": item[variant]["report"]["actor_decisions"],
                               "scenario_schedule_sha256": item[variant]["report"]["scenario_schedule_sha256"],
                               "initial_role_sha256": item[variant]["report"]["initial_role_sha256"],
                               "selected_role_sha256": item[variant]["report"]["selected_role_sha256"],
                               "actor_weight_delta_l2": item[variant]["report"][
                                   "actor_weight_delta_l2"],
                               "actor_weight_changed_parameter_count": item[variant]["report"][
                                   "actor_weight_changed_parameter_count"],
                               "both_roles_changed": all(
                                   item[variant]["report"]["selected_role_sha256"][role]
                                   != item[variant]["report"]["initial_role_sha256"][role]
                                   for role in AGENTS),
                               "feedback_rows": item[variant]["report"]["feedback_rows"],
                               "selected_feedback_rows": item[variant]["report"]["selected_feedback_rows"],
                               "feedback_window_actor_rows": item[variant]["report"][
                                   "feedback_window_actor_rows"],
                               "zero_action_coverage_rows": item[variant]["report"][
                                   "zero_action_coverage_rows"],
                               "physical_no_legal_action_rows": item[variant]["report"][
                                   "physical_no_legal_action_rows"],
                               "post_policy_interventions": item[variant]["report"]["post_policy_interventions"],
                               "checkpoint_sha256": sha256((output_dir / "_warehouse_dual_v5_work" /
                                   f"pair_{item['seed']}_{variant}_matched.pt").read_bytes()).hexdigest(),
                           } for variant in ("raw", "regularized")}}
                      for item in candidates],
                  "raw_program_feedback_rows": selected["raw"]["report"]["feedback_rows"],
                  "regularized_program_feedback_rows": selected["regularized"]["report"]["feedback_rows"],
                  "zero_action_coverage_rows": {
                      variant: selected[variant]["report"]["zero_action_coverage_rows"]
                      for variant in ("raw", "regularized")},
                  "zero_action_coverage_definition": ZERO_ACTION_COVERAGE_DEFINITION,
                  "selected_regularized_feedback_rows": selected["regularized"]["report"]["selected_feedback_rows"],
                  "post_policy_override_count": 0,
                  "joint_steps": {variant: selected[variant]["report"]["joint_steps"]
                                  for variant in ("raw", "regularized")},
                  "development": [{"pair_seed": item["seed"],
                                   "raw": item["raw"]["report"]["selected_development"],
                                   "regularized": item["regularized"]["report"]["selected_development"],
                                   "preliminary_program_audit": item["selected_program_audit"],
                                   "selected_program_audit": (final_program_audit
                                       if item["seed"] == selected["seed"] else {}),
                                   "regularized_feedback_rows": item["regularized"]["report"]["feedback_rows"]}
                                  for item in candidates],
                  "heldout": heldout,
                  "artifact_sha256": {v: artifacts[v]["sha256"] for v in artifacts},
                  "role_sha256": {v: artifacts[v]["role_sha256"] for v in artifacts},
                  "training_report_sha256": {v: reports[v]["sha256"] for v in reports},
                  "program_sha256": program_hashes}
    _write_json(receipt_path, acceptance)
    if install:
        _install_accepted_assets(acceptance, output_dir)
    return acceptance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-seeds", type=int, nargs=3, default=DEFAULT_SEEDS)
    parser.add_argument("--joint-steps", type=int, default=FULL_JOINT_STEPS)
    parser.add_argument("--bc-episodes", type=int, default=200)
    parser.add_argument("--bc-epochs", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/deployment")
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--enable-heldout", action="store_true")
    args = parser.parse_args()
    result = run_experiment(pair_seeds=tuple(args.pair_seeds), joint_steps=args.joint_steps,
                            bc_episodes=args.bc_episodes, bc_epochs=args.bc_epochs,
                            output_dir=args.output_dir, install=args.install,
                            enable_heldout=args.enable_heldout)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"development", "heldout"}}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
