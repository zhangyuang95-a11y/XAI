"""One parameterized, paired Warehouse v6 BC + constrained PPO runner.

This module trains a raw and a regularized dual Actor from the same offline
BC tensors and the same 120-step scenario schedule. During self-play, both
actions are sampled from their own NN logits. The offline BC teacher and
programs never enter the environment action path.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import argparse
import base64
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from experiments.warehouse.protocol import (
    FINAL_PROGRAM_SPLIT_COUNTS, REGISTERED_BC_RECIPE_VERSION, REGISTERED_BC_SPEC,
    REGISTERED_BC_SPEC_SHA256, PROGRAM_FIT_METHOD_CANDIDATES,
    REGISTERED_RISK_DUAL_CONFIG, REGISTERED_RISK_DUAL_CONFIG_SHA256,
    bc_data_seed_schedule, bc_critic_reset_schedule, bc_report_valid,
    SCENARIO_BLOCK_STEPS, online_program_cohorts_valid,
    scenario_seed, scenario_schedule_sha256,
)
from env.warehouse.decision_protocol import canonical_sha256, independent_agent_seed
from env.warehouse.domain import participant_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import AGENTS, action_masks
from env.warehouse.public_observation_v6 import (
    FEATURE_SCHEMA_SHA256, OBSERVATION_DIM, OBSERVATION_VERSION,
    OBSERVATION_FEATURE_NAMES,
    encode_public_observation, public_pre_action_snapshot,
    rebuild_public_pre_action_state,
)
from .warehouse_v6_actor import (
    MODEL_VERSION, FrozenWarehouseV6Actor, WarehouseV6Actor, export_actor,
    policy_features_torch, role_hashes,
)
from .warehouse_v6_program import (
    ACTOR_SPARSE_TOPOLOGY_METHOD, AXIS_TREE_METHOD, WarehouseV6Program,
    fit_program, audit_frozen_program, current_actor_fidelity,
)
from .warehouse_v6_shutdown_cost import (
    RiskDualConfig, ShutdownRiskDual, shutdown_cost_gae,
    joint_shutdown_surrogate, episode_balanced_cost_critic_loss,
)
from .warehouse_v6_teacher_lookahead import public_lookahead_teacher


RUNNER_VERSION = "warehouse-v6-paired-complete-event-cost-ppo.v8-candidate-protocol"
PROGRAM_LAMBDA = 0.01
PROGRAM_START_FRACTION = 0.80
PROGRAM_RAMP_FRACTION = 0.05
PROGRAM_BUFFER_STATES = 20_000
PROGRAM_SPLIT_EPISODES = (50, 25, 25, 24)
PROGRAM_MIN_EPISODES = sum(PROGRAM_SPLIT_EPISODES)
# PPO's training score must rank a threshold-quality delivery trajectory
# above the inert 120-WAIT trajectory. These values affect training rewards
# only; the participant environment and acceptance metrics are untouched.
TRAINING_REWARD_OVERRIDES = {"delivery_points": 100.0,
                             "robot_collision_points": -50.0,
                             "shutdown_points": -200.0}
DEFAULT_BC_EPISODES = REGISTERED_BC_SPEC["bc_episodes"]
DEFAULT_BC_EPOCHS = REGISTERED_BC_SPEC["bc_epochs"]
DEFAULT_DAGGER_ROUNDS = REGISTERED_BC_SPEC["dagger_rounds"]
DEFAULT_DAGGER_EPISODES = REGISTERED_BC_SPEC["dagger_episodes"]
DEFAULT_DAGGER_EPOCHS = REGISTERED_BC_SPEC["dagger_epochs"]

# A single, replayable offline BC recipe is completed before either PPO
# variant starts. Every data seed and stage is fixed by the pair seed. The
# offline teacher and learned collision critic are absent from execution.
BC_RECIPE_VERSION = REGISTERED_BC_RECIPE_VERSION
BC_INITIAL_TEACHER_EPISODES = REGISTERED_BC_SPEC["bc_episodes"]
BC_INITIAL_EPOCHS = REGISTERED_BC_SPEC["bc_epochs"]
BC_INITIAL_DAGGER_ROUNDS = REGISTERED_BC_SPEC["dagger_rounds"]
BC_INITIAL_DAGGER_EPISODES = REGISTERED_BC_SPEC["dagger_episodes"]
BC_INITIAL_DAGGER_EPOCHS = REGISTERED_BC_SPEC["dagger_epochs"]
BC_RISK_TEACHER_EPISODES = REGISTERED_BC_SPEC["risk_teacher_episodes"]
BC_RISK_VISIT_EPISODES = REGISTERED_BC_SPEC["risk_visit_episodes"]
BC_LARGE_TEACHER_EPISODES = REGISTERED_BC_SPEC["large_teacher_episodes"]
BC_LARGE_VISIT_EPISODES = REGISTERED_BC_SPEC["large_visit_episodes"]
BC_DEEP_TEACHER_EPISODES = REGISTERED_BC_SPEC["deep_teacher_episodes"]
BC_DEEP_DAGGER_ROUNDS = REGISTERED_BC_SPEC["deep_dagger_rounds"]
BC_DEEP_DAGGER_EPISODES = REGISTERED_BC_SPEC["deep_dagger_episodes"]
BC_RISK_EPOCHS = REGISTERED_BC_SPEC["risk_epochs"]
BC_DEEP_EPOCHS = REGISTERED_BC_SPEC["deep_epochs"]
BC_BATCH_SIZE = REGISTERED_BC_SPEC["batch_size"]
BC_RISK_WEIGHT = REGISTERED_BC_SPEC["risk_weight"]
BC_MARGIN_GAP = REGISTERED_BC_SPEC["margin_gap"]
BC_MARGIN_WEIGHT = REGISTERED_BC_SPEC["margin_weight"]
BC_NEAR_INCREMENT = REGISTERED_BC_SPEC["near_increment"]
BC_LOW_RETURN_INCREMENT = REGISTERED_BC_SPEC["low_return_increment"]
BC_SAMPLED_COLLISION_INCREMENT = REGISTERED_BC_SPEC["sampled_collision_increment"]
BC_ARGMAX_COLLISION_INCREMENT = REGISTERED_BC_SPEC["argmax_collision_increment"]
BC_DUPLICATE_COLLISION_CAP = REGISTERED_BC_SPEC["duplicate_collision_cap"]

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_FILES = (
    *(path.relative_to(_SOURCE_ROOT).as_posix()
      for path in sorted((_SOURCE_ROOT / "env" / "warehouse").glob("*.py"))),
    *(path.relative_to(_SOURCE_ROOT).as_posix()
      for path in sorted((_SOURCE_ROOT / "env" / "warehouse_legacy_v3").glob("*.py"))),
    "env/warehouse_legacy_v3/source_manifest.json",
    "backend/__init__.py", "backend/training/__init__.py",
    "backend/training/legacy/__init__.py",
    "backend/training/legacy/warehouse_v5/__init__.py",
    "backend/training/legacy/warehouse_v5/warehouse_selective_v5.py",
    "backend/training/legacy/warehouse_v5/warehouse_sparse_v5.py",
    "backend/training/warehouse_selective_v5.py",
    "backend/training/warehouse_sparse_v5.py",
    "backend/inference/__init__.py", "backend/inference/warehouse_structured_v5.py",
    "backend/training/warehouse_v6_actor.py", "backend/training/warehouse_v6_actor_training.py",
    "backend/training/warehouse_v6_collision_critic.py", "backend/training/warehouse_v6_core_tree.py",
    "backend/training/warehouse_v6_program.py",
    "backend/training/warehouse_v6_runner.py", "backend/training/warehouse_v6_shutdown_cost.py",
    "backend/training/warehouse_v6_teacher.py", "backend/training/warehouse_v6_teacher_lookahead.py",
    "core/__init__.py", "core/policy_contracts.py", "core/policy_program_regularizer.py",
    "core/program.py", "core/rcpd.py", "core/rcpd_config.py", "core/rcpd_tree.py",
    "env/__init__.py", "domains/warehouse/__init__.py",
    "experiments/__init__.py", "experiments/warehouse/__init__.py",
    "experiments/warehouse/protocol.py", "experiments/warehouse/admission.py",
    "experiments/warehouse/acceptance.py", "experiments/warehouse/quality_replay.py",
    "domains/warehouse/neural_turnbased.py", "domains/warehouse/turnbased.py",
    "study_v3/policies.py", "study_v3/registry.py", "study_v3/release_catalog.py",
    "study_v3/store.py", "study_v3/qa.py", "study_v3/domain_contracts.py", "study_v3/__init__.py",
    "scripts/package_warehouse_v6_release.py", "scripts/start_warehouse_v6_candidate_release.py",
    "configs/study_v3_warehouse.json", "configs/study_v3_warehouse_neural.json",
    "configs/study_v3_warehouse_neural_v5.json", "configs/study_v3_warehouse_neural_v6.json",
    "requirements-warehouse-v6-training.txt",
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


def _training_config_sha256(spec: Mapping) -> str:
    """Include BC recipe and source closure for exact checkpoint resumption."""
    return _digest_json({key: value for key, value in spec.items()
                         if key != "output_dir"})


def _scientific_run_spec_sha256(spec: Mapping) -> str:
    """Use the preregistered PairRunSpec identity in scientific receipts."""
    if spec.get("smoke"):
        return _training_config_sha256(spec)
    from experiments.warehouse.protocol import PairRunSpec
    registered = PairRunSpec(
        candidate_id=spec["candidate_id"], pair_seed=spec["pair_seed"],
        joint_steps=spec["joint_steps"], output_dir=spec["output_dir"],
        release_id=spec["release_id"],
        scenario_namespace=spec["scenario_namespace"],
        checkpoint_seeds=tuple(spec["checkpoint_seeds"]),
        program_fit_method=spec["program_fit_method"],
        bc_recipe_sha256=spec["bc_recipe_sha256"],
        risk_dual_config_sha256=spec["risk_dual_config_sha256"],
        bc_data_seed_schedule_sha256=spec["bc_data_seed_schedule_sha256"],
        bc_critic_reset_schedule_sha256=spec[
            "bc_critic_reset_schedule_sha256"],
    )
    canonical = registered.to_mapping()
    if _digest_json({key: spec.get(key) for key in canonical}) != _digest_json(canonical):
        raise RuntimeError("Warehouse canonical run spec or schedule drifted")
    if _training_config_sha256(spec) != registered.training_config_sha256(
            spec["source_closure_sha256"]):
        raise RuntimeError("Warehouse registered BC config hash drifted")
    return registered.sha256()


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
            REGISTERED_ARCHITECTURE_MODE, REGISTERED_BC_RECIPE_VERSION,
            REGISTERED_BC_SPEC_SHA256, PROGRAM_FIT_METHOD_CANDIDATES,
            REGISTERED_RISK_DUAL_CONFIG_SHA256,
            final_program_fit_seed_schedule, online_program_extraction_steps,
        )
        if result["joint_steps"] not in BUDGETS:
            raise ValueError("Warehouse v6 budget is not preregistered")
        if result["scenario_namespace"] != "warehouse-v6-matched-120-step-blocks":
            raise ValueError("Warehouse v6 scenario namespace is not preregistered")
        if result["pair_seed"] in DEVELOPMENT_PAIR_SEEDS:
            expected_seeds = DEVELOPMENT_SELECTION_SEEDS
        elif result["pair_seed"] in FORMAL_PAIR_SEEDS:
            expected_seeds = FORMAL_SELECTION_SEEDS
        else:
            raise ValueError("Warehouse v6 pair seed is not preregistered")
        if tuple(seeds) != tuple(expected_seeds):
            raise ValueError("Warehouse v6 checkpoint seeds do not match registered block")
        expected_extractions = online_program_extraction_steps(result["joint_steps"])
        requested_extractions = run_spec.get("online_program_extraction_steps",
                                             expected_extractions)
        if (run_spec.get("architecture_mode", REGISTERED_ARCHITECTURE_MODE) !=
                REGISTERED_ARCHITECTURE_MODE
                or run_spec.get("program_fit_method") not in
                PROGRAM_FIT_METHOD_CANDIDATES
                or run_spec.get("bc_recipe_version", REGISTERED_BC_RECIPE_VERSION) !=
                REGISTERED_BC_RECIPE_VERSION
                or run_spec.get("bc_recipe_sha256") != REGISTERED_BC_SPEC_SHA256
                or run_spec.get("risk_dual_config_sha256") !=
                REGISTERED_RISK_DUAL_CONFIG_SHA256
                or not isinstance(requested_extractions, (list, tuple))
                or tuple(requested_extractions) != expected_extractions
                or any(type(step) is not int for step in requested_extractions)):
            raise ValueError("Warehouse v6 registered architecture, BC, or extraction schedule mismatch")
        result["architecture_mode"] = REGISTERED_ARCHITECTURE_MODE
        result["program_fit_method"] = run_spec["program_fit_method"]
        result["bc_recipe_version"] = REGISTERED_BC_RECIPE_VERSION
        result["bc_recipe_sha256"] = REGISTERED_BC_SPEC_SHA256
        result["risk_dual_config_sha256"] = REGISTERED_RISK_DUAL_CONFIG_SHA256
        result["online_program_extraction_steps"] = list(expected_extractions)
        release_id = run_spec.get("release_id")
        if not isinstance(release_id, str):
            raise ValueError("Warehouse v6 registered run requires release ID")
        schedule = final_program_fit_seed_schedule(release_id, result["pair_seed"])
        if (run_spec.get("final_program_fit_seed_schedule") != schedule
                or run_spec.get("final_program_fit_seed_schedule_sha256") !=
                _digest_json(schedule)):
            raise ValueError("Warehouse v6 final program schedule is not preregistered")
        result["release_id"] = release_id
        result["final_program_fit_seed_schedule"] = schedule
        result["final_program_fit_seed_schedule_sha256"] = _digest_json(schedule)
        bc_schedule = bc_data_seed_schedule(result["pair_seed"])
        if (run_spec.get("bc_data_seed_schedule") != bc_schedule
                or run_spec.get("bc_data_seed_schedule_sha256") !=
                _digest_json(bc_schedule)):
            raise ValueError("Warehouse v6 BC stage seed schedule is not preregistered")
        result["bc_data_seed_schedule"] = bc_schedule
        result["bc_data_seed_schedule_sha256"] = _digest_json(bc_schedule)
        critic_resets = bc_critic_reset_schedule(result["pair_seed"], release_id)
        if (run_spec.get("bc_critic_reset_schedule") != critic_resets
                or run_spec.get("bc_critic_reset_schedule_sha256") !=
                _digest_json(critic_resets)):
            raise ValueError("Warehouse v6 critic reset schedule is not preregistered")
        result["bc_critic_reset_schedule"] = critic_resets
        result["bc_critic_reset_schedule_sha256"] = _digest_json(critic_resets)
    if smoke:
        result["smoke"] = True
    elif any(key in run_spec for key in REGISTERED_BC_SPEC):
        raise ValueError("registered Warehouse v6 BC recipe cannot be overridden")
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
    if not smoke:
        result.update(REGISTERED_BC_SPEC)
        if (not set(PROGRAM_FIT_METHOD_CANDIDATES) <=
                {ACTOR_SPARSE_TOPOLOGY_METHOD}):
            raise RuntimeError("Warehouse registered program implementation drifted")
        if (asdict(RiskDualConfig()) != REGISTERED_RISK_DUAL_CONFIG
                or _digest_json(asdict(RiskDualConfig())) !=
                REGISTERED_RISK_DUAL_CONFIG_SHA256):
            raise RuntimeError("Warehouse PPO shutdown risk-dual config drifted")
    closure = source_closure_sha256()
    if ("source_closure_sha256" in run_spec
            and run_spec["source_closure_sha256"] != closure):
        raise ValueError("Warehouse v6 run spec source closure mismatch")
    result["source_closure_sha256"] = closure
    return result


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
            chosen = public_lookahead_teacher(obs, mask)
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
        for indices in torch.randperm(len(obs), generator=generator).split(
                REGISTERED_BC_SPEC["initial_bc_batch_size"]):
            logits = model.outputs_encoded(encoded[indices], mask[indices])
            loss = nn.functional.cross_entropy(logits.reshape(-1, len(ACTIONS)),
                                               labels[indices].reshape(-1))
            tree_loss, gate_supervision = [], []
            for index, role in enumerate(AGENTS):
                role_rows = encoded[indices, index]
                actor = model.actors[role]
                tree_logits, _ = actor.tree(role_rows)
                tree_loss.append(nn.functional.cross_entropy(
                    tree_logits.masked_fill(mask[indices, index] <= 0.5, -1e9),
                    labels[indices, index]))
                with torch.no_grad():
                    tree_prob = torch.softmax(tree_logits.masked_fill(
                        mask[indices, index] <= 0.5, -1e9), -1)
                    easy = (tree_prob.argmax(-1) == labels[indices, index]) & (
                        tree_prob.max(-1).values >= 0.9)
                gate_target = torch.where(easy, 0.1, 0.9).to(role_rows.dtype)
                gate_supervision.append(nn.functional.binary_cross_entropy(
                    actor.gate_values(role_rows), gate_target))
            loss = loss + REGISTERED_BC_SPEC["initial_tree_loss_weight"] * \
                torch.stack(tree_loss).mean()
            # The gate learns when the public-state tree already predicts the
            # offline label and when the relation residual must take over.
            # Only BC/DAgger training rows supply this target, never program
            # qualification or independent evaluation episodes.
            loss = loss + REGISTERED_BC_SPEC["initial_gate_loss_weight"] * \
                torch.stack(gate_supervision).mean()
            # Both variants inherit the same learned preference for compact
            # residual corrections. Raw receives no program gradient later.
            loss = loss + REGISTERED_BC_SPEC["initial_bound_loss_weight"] * torch.stack([
                model.actors[role].bounds().mean() for role in AGENTS]).mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.actors.parameters(),
                                     REGISTERED_BC_SPEC["initial_bc_gradient_clip"])
            optimizer.step()
    with torch.no_grad():
        predicted = model.outputs_encoded(encoded, mask).argmax(-1)
        return float((predicted == labels).float().mean())


def _bc_reliability(model: WarehouseV6Actor, obs: torch.Tensor,
                    mask: torch.Tensor, labels: torch.Tensor) -> dict:
    """Track sampled dual-Actor reliability, including public hard states."""
    if not len(obs):
        raise ValueError("Warehouse v6 BC reliability requires rows")
    probabilities, correct = [], []
    with torch.no_grad():
        for begin in range(0, len(obs), 512):
            end = min(begin + 512, len(obs))
            logits = model.outputs(obs[begin:end], mask[begin:end])
            probabilities.append(torch.softmax(logits, -1).gather(
                -1, labels[begin:end, :, None]).squeeze(-1))
            correct.append(logits.argmax(-1) == labels[begin:end])
    p = torch.cat(probabilities)
    accurate = torch.cat(correct)
    distance_column = OBSERVATION_FEATURE_NAMES.index("robots_shortest_distance")
    margin_column = OBSERVATION_FEATURE_NAMES.index("self_charger_return_margin")
    near = obs[:, 0, distance_column] * 42.0 <= 2.0
    low_margin = obs[:, :, margin_column].min(-1).values * 50.0 <= 4.0

    def describe(selected: torch.Tensor) -> dict:
        if not bool(selected.any()):
            return {"rows": 0, "role_teacher_action_probability": None,
                    "joint_teacher_action_probability": None,
                    "role_teacher_top1_accuracy": None}
        return {"rows": int(selected.sum()),
                "role_teacher_action_probability": {
                    role: float(p[selected, index].mean())
                    for index, role in enumerate(AGENTS)},
                "joint_teacher_action_probability": float(p[selected].prod(-1).mean()),
                "role_teacher_top1_accuracy": {
                    role: float(accurate[selected, index].float().mean())
                    for index, role in enumerate(AGENTS)}}

    all_rows = torch.ones(len(obs), dtype=torch.bool)
    return {"all": describe(all_rows), "near_robots": describe(near),
            "low_return_margin": describe(low_margin),
            "near_and_low_margin": describe(near & low_margin)}


def _initialize_public_trees(model: WarehouseV6Actor, obs: torch.Tensor,
                             mask: torch.Tensor, labels: torch.Tensor) -> dict:
    """Fit both sparse Actor branches before residual BC from public rows."""
    import sklearn
    from sklearn.tree import DecisionTreeClassifier

    features = policy_features_torch(obs, mask).numpy()
    reports = {}
    for index, role in enumerate(AGENTS):
        classifier = DecisionTreeClassifier(
            max_depth=5, max_leaf_nodes=16, min_samples_leaf=32,
            random_state=19 + index)
        classifier.fit(features[:, index], labels[:, index].numpy())
        detail = model.actors[role].initialize_tree_from_classifier(classifier)
        with torch.no_grad():
            tree_logits, path = model.actors[role].tree(
                torch.from_numpy(features[:, index]))
            decision = tree_logits.masked_fill(mask[:, index] <= 0.5,
                                               -1e9).argmax(-1)
        reports[role] = {**detail,
                         "sklearn_version": sklearn.__version__,
                         "fitting_rows": len(features),
                         "split_frozen_during_bc_and_ppo": False,
                         "split_training_method": "exact_hard_forward_sparse_STE",
                         "tree_initialization_sha256": _digest_json({
                             "split_weight": model.actors[role].tree.split.weight.detach().tolist(),
                             "split_bias": model.actors[role].tree.split.bias.detach().tolist(),
                             "left_refs": model.actors[role].tree.left_refs.detach().tolist(),
                             "right_refs": model.actors[role].tree.right_refs.detach().tolist(),
                             "node_count": model.actors[role].tree.node_count.detach().tolist(),
                             "leaf_count": model.actors[role].tree.leaf_count.detach().tolist(),
                             "leaf_logits": model.actors[role].tree.leaf_logits.detach().tolist(),
                         }),
                         "training_leaf_occupancy": int((path.sum(0) > 0).sum()),
                         "training_label_accuracy": float((
                             decision == labels[:, index]).float().mean())}
    return reports


def _pretrain_pair_smoke(model: WarehouseV6Actor, *, pair_seed: int,
                         spec: Mapping) -> dict:
    """Small legacy BC path used only by the explicit trainer smoke test."""
    torch.manual_seed(pair_seed)
    initial_hashes = role_hashes(model)
    obs, mask, labels, teacher = _teacher_rows(spec["bc_episodes"], pair_seed + 10000)
    tree_initialization = _initialize_public_trees(model, obs, mask, labels)
    optimizer = torch.optim.Adam(model.actors.parameters(), lr=3e-4)
    generator = torch.Generator().manual_seed(pair_seed + 31)
    accuracy = _bc_epochs(model, optimizer, obs, mask, labels,
                          epochs=spec["bc_epochs"], generator=generator)
    initial_reliability = _bc_reliability(model, obs, mask, labels)
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
                label = public_lookahead_teacher(row_obs, row_mask)
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
        visited_reliability = _bc_reliability(
            model, all_obs[-1], all_mask[-1], all_labels[-1])
        mean = np.mean(np.asarray(outcomes, dtype=np.float64), axis=0)
        history.append({"round": round_number, "visited_episodes": spec["dagger_episodes"],
                        "visited_joint_states": len(visited_obs),
                        "public_label_accuracy": accuracy,
                        "visited_sampled_action_reliability": visited_reliability,
                        "nn_mean_deliveries": float(mean[0]),
                        "nn_mean_collisions": float(mean[1]),
                        "nn_shutdown_episode_rate": float(mean[2])})
    return {"source": "public_observation_only_offline_bc_and_nn_visits",
            "observation_version": OBSERVATION_VERSION,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "bc_fit_episode_seeds": list(range(pair_seed + 10000,
                                               pair_seed + 10000 + spec["bc_episodes"])),
            "training_dependencies": {"torch": torch.__version__,
                                      "numpy": np.__version__,
                                      "sklearn": tree_initialization[AGENTS[0]]["sklearn_version"]},
            "teacher": teacher, "initial_role_sha256": initial_hashes,
            "tree_initialization": tree_initialization,
            "initial_bc_sampled_action_reliability": initial_reliability,
            "final_bc_training_reliability": _bc_reliability(
                model, torch.cat(all_obs), torch.cat(all_mask),
                torch.cat(all_labels)),
            "pretrained_role_sha256": role_hashes(model),
            "public_label_accuracy": accuracy, "dagger": history,
            "teacher_uses_hidden_goal_or_route_state": False}


def _bc_stage_seeds(pair_seed: int, stage: int, count: int) -> list[int]:
    """Allocate disjoint public-scenario seeds outside every protocol block."""
    if pair_seed < 0 or stage < 1 or stage > 90 or count < 1 or count > 999:
        raise ValueError("invalid Warehouse BC stage seed schedule")
    return [pair_seed * 100_000 + stage * 1_000 + index
            for index in range(count)]


def _bc_rows_sha256(rows: tuple[torch.Tensor, ...]) -> str:
    digest = sha256()
    for item in rows:
        array = item.detach().contiguous().cpu().numpy()
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _collect_bc_visits(model: WarehouseV6Actor, seeds: Sequence[int], *,
                       deterministic: bool, deduplicate_collisions: bool,
                       collision_state_counts: dict[bytes, int] | None = None
                       ) -> tuple[tuple[torch.Tensor, ...], dict]:
    """Visit using only the current two NN outputs, then label public states.

    Argmax visits are a training-data intervention. Frozen execution and PPO
    always use their specified NN sampling path. Neither public teacher labels
    nor collision de-duplication choose a current environment action.
    """
    actor = FrozenWarehouseV6Actor.from_model(model)
    obs_rows, mask_rows, labels, hard_rows = [], [], [], []
    outcomes, joint_teacher_probability = [], []
    repeated_collision_frames = cross_episode_skips = 0
    collision_rows_kept = 0
    counts = collision_state_counts if collision_state_counts is not None else {}
    time_columns = [OBSERVATION_FEATURE_NAMES.index(name) for name in (
        "elapsed_fraction", "self_steps_since_charging",
        "partner_steps_since_charging")]
    for seed in seeds:
        env = _environment(training_reward=False)
        env.reset(seed=int(seed))
        _set_dual_nn_mode(env)
        prior_collision_key, repeat_streak = None, 0
        while not (env.state.terminated or env.state.truncated):
            obs, mask = encode_public_observation(env.state), action_masks(env.state)
            label = public_lookahead_teacher(obs, mask)
            positions = tuple(env.state.by_id(role).position for role in AGENTS)
            actions, distributions = actor.act(
                env.state, base_seed=int(seed), deterministic=deterministic)
            joint_teacher_probability.append(float(np.prod([
                distributions[role].probabilities[int(label[index])]
                for index, role in enumerate(AGENTS)])))
            before = env.state.robot_collision_events
            _, _, _, _, info = env.step(actions)
            if info["proposed_actions"] != actions or info["action_override_agents"]:
                raise RuntimeError("Warehouse BC NN visit rewrote a proposed action")
            collided = env.state.robot_collision_events > before
            key = (positions, tuple(actions[role] for role in AGENTS))
            repeat_streak = repeat_streak + 1 if collided and key == prior_collision_key else 0
            prior_collision_key = key if collided else None
            repeated_collision_frames += int(collided and repeat_streak >= 4)
            keep = (not deduplicate_collisions or not deterministic or not collided
                    or repeat_streak < 4 or repeat_streak % 10 == 0)
            if keep and collided and deduplicate_collisions:
                signature_obs = np.round(obs.copy(), 3)
                signature_obs[:, time_columns] = 0
                signature = signature_obs.tobytes() + label.tobytes()
                if counts.get(signature, 0) >= BC_DUPLICATE_COLLISION_CAP:
                    keep = False
                    cross_episode_skips += 1
                else:
                    counts[signature] = counts.get(signature, 0) + 1
            if keep:
                obs_rows.append(obs)
                mask_rows.append(mask)
                labels.append(label)
                hard_rows.append(
                    BC_ARGMAX_COLLISION_INCREMENT if collided and deterministic
                    else BC_SAMPLED_COLLISION_INCREMENT if collided else 0.0)
                collision_rows_kept += int(collided)
        outcomes.append((env.state.total_deliveries,
                         env.state.robot_collision_events,
                         bool(env.state.shutdown_count)))
    rows = (torch.from_numpy(np.stack(obs_rows)),
            torch.from_numpy(np.stack(mask_rows)),
            torch.from_numpy(np.stack(labels)).long(),
            torch.tensor(hard_rows, dtype=torch.float32))
    mean = np.mean(np.asarray(outcomes, dtype=np.float64), axis=0)
    return rows, {
        "episode_seeds": list(map(int, seeds)), "episodes": len(seeds),
        "rows_kept_for_training": len(obs_rows),
        "data_sha256": _bc_rows_sha256(rows),
        "training_only_argmax_visits": bool(deterministic),
        "mean_deliveries": float(mean[0]), "mean_collisions": float(mean[1]),
        "shutdown_episode_rate": float(mean[2]),
        "joint_teacher_action_probability": float(np.mean(joint_teacher_probability)),
        "repeated_collision_frames_deduplicated": repeated_collision_frames,
        "cross_episode_collision_rows_skipped": cross_episode_skips,
        "max_collision_signature_rows": max(counts.values(), default=0),
        "collision_rows_kept": collision_rows_kept,
        "post_policy_interventions": 0,
    }


def _risk_bc_epochs(model: WarehouseV6Actor, critic: nn.Module,
                    parts: Sequence[tuple[torch.Tensor, ...]], *, epochs: int,
                    generator_seed: int, margin_weight: float,
                    near_increment: float, low_return_increment: float) -> dict:
    """Offline NN imitation with a separately learned real-event risk critic."""
    from .warehouse_v6_collision_critic import risk_matrix
    obs, mask, labels, hard = (
        torch.cat([part[index] for part in parts]) for index in range(4))
    encoded = policy_features_torch(obs, mask)
    risk = risk_matrix(critic, obs, mask)
    dist_column = OBSERVATION_FEATURE_NAMES.index("robots_shortest_distance")
    margin_column = OBSERVATION_FEATURE_NAMES.index("self_charger_return_margin")
    near = obs[:, 0, dist_column] * 42.0 <= 2.0
    remaining = obs[:, :, margin_column] * 50.0
    low = (remaining >= 0.0) & (remaining <= 6.0)
    weights = (1.0 + near_increment * near.float()[:, None]
               + low_return_increment * low.float() + hard[:, None])
    weights /= weights.mean()
    optimizer = torch.optim.Adam(
        model.actors.parameters(), lr=REGISTERED_BC_SPEC["risk_bc_learning_rate"])
    generator = torch.Generator().manual_seed(generator_seed)
    last = {}
    for epoch in range(epochs):
        for indices in torch.randperm(len(obs), generator=generator).split(BC_BATCH_SIZE):
            logits = model.outputs_encoded(encoded[indices], mask[indices])
            p = torch.softmax(logits, -1)
            label = labels[indices]
            ce = nn.functional.cross_entropy(
                logits.reshape(-1, len(ACTIONS)), label.reshape(-1),
                reduction="none").reshape(-1, len(AGENTS))
            chosen = logits.gather(-1, label[..., None]).squeeze(-1)
            other = logits.masked_fill(
                nn.functional.one_hot(label, len(ACTIONS)).bool(), -1e9
            ).max(-1).values
            margin = (nn.functional.relu(BC_MARGIN_GAP + other - chosen).square()
                      * weights[indices]).mean()
            expected_collision = (
                p[:, 0, :, None] * p[:, 1, None, :] * risk[indices]
            ).sum((1, 2)).mean()
            loss = ((ce * weights[indices]).mean() + margin_weight * margin
                    + BC_RISK_WEIGHT * expected_collision)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.actors.parameters(),
                                     REGISTERED_BC_SPEC["risk_bc_gradient_clip"])
            optimizer.step()
        last = {"epoch": epoch + 1,
                "mean_batch_ce": float(ce.detach().mean()),
                "mean_batch_margin": float(margin.detach()),
                "mean_batch_expected_collision": float(expected_collision.detach())}
    return {"rows": len(obs), "epochs": epochs, "batch_size": BC_BATCH_SIZE,
            "data_sha256": _bc_rows_sha256((obs, mask, labels, hard)),
            "learning_rate": REGISTERED_BC_SPEC["risk_bc_learning_rate"],
            "gradient_clip": REGISTERED_BC_SPEC["risk_bc_gradient_clip"],
            "near_fraction": float(near.float().mean()),
            "low_return_margin_fraction": float(low.float().mean()),
            "collision_hard_fraction": float((hard > 0).float().mean()),
            "risk_weight": BC_RISK_WEIGHT, "margin_gap": BC_MARGIN_GAP,
            "margin_weight": margin_weight,
            "near_increment": near_increment,
            "low_return_increment": low_return_increment,
            "end": last}


def _pretrain_pair_production(model: WarehouseV6Actor, *, pair_seed: int,
                              spec: Mapping) -> dict:
    """Rebuild the complete high-reliability public BC start from scratch.

    Every phase is deterministic from the immutable pair seed. The BC teacher
    labels stored pre-action states; both rollout actions come from the
    current dual NN. The learned critic receives true env.step collision
    events and supplies gradients only before PPO self-play begins.
    """
    from .warehouse_v6_collision_critic import train_collision_critic

    torch.set_num_threads(2)
    torch.manual_seed(pair_seed)
    expected_seed_blocks = bc_data_seed_schedule(pair_seed)
    expected_critic_resets = bc_critic_reset_schedule(
        pair_seed, spec["release_id"])
    if (spec.get("bc_recipe_version") != BC_RECIPE_VERSION
            or spec.get("bc_recipe_sha256") != REGISTERED_BC_SPEC_SHA256
            or spec.get("bc_data_seed_schedule") != expected_seed_blocks
            or spec.get("bc_data_seed_schedule_sha256") !=
            _digest_json(expected_seed_blocks)
            or spec.get("bc_critic_reset_schedule") != expected_critic_resets
            or spec.get("bc_critic_reset_schedule_sha256") !=
            _digest_json(expected_critic_resets)
            or spec.get("risk_dual_config_sha256") !=
            REGISTERED_RISK_DUAL_CONFIG_SHA256):
        raise RuntimeError("Warehouse production BC recipe or stage seed lock drifted")
    initial_hashes = role_hashes(model)
    stages: list[dict] = []
    all_seed_blocks: dict[str, list[int]] = {}
    initial_fit_config = {
        "learning_rate": REGISTERED_BC_SPEC["initial_bc_learning_rate"],
        "batch_size": REGISTERED_BC_SPEC["initial_bc_batch_size"],
        "tree_loss_weight": REGISTERED_BC_SPEC["initial_tree_loss_weight"],
        "gate_loss_weight": REGISTERED_BC_SPEC["initial_gate_loss_weight"],
        "bound_loss_weight": REGISTERED_BC_SPEC["initial_bound_loss_weight"],
        "gradient_clip": REGISTERED_BC_SPEC["initial_bc_gradient_clip"],
    }

    def teacher_stage(name: str, stage: int, count: int
                      ) -> tuple[torch.Tensor, ...]:
        seeds = _bc_stage_seeds(pair_seed, stage, count)
        if expected_seed_blocks.get(name) != seeds:
            raise RuntimeError("Warehouse public BC teacher stage seed drifted")
        all_seed_blocks[name] = seeds
        obs, mask, labels, summary = _teacher_rows(count, seeds[0])
        rows = (obs, mask, labels, torch.zeros(len(obs), dtype=torch.float32))
        stages.append({"stage": name, "kind": "public_teacher",
                       "episode_seeds": seeds, "rows": len(obs),
                       "data_sha256": _bc_rows_sha256(rows), **summary})
        return rows

    def visit_stage(name: str, stage: int, count: int, *,
                    deterministic: bool = False,
                    collision_counts: dict[bytes, int] | None = None
                    ) -> tuple[torch.Tensor, ...]:
        seeds = _bc_stage_seeds(pair_seed, stage, count)
        if expected_seed_blocks.get(name) != seeds:
            raise RuntimeError("Warehouse dual-NN BC visit stage seed drifted")
        all_seed_blocks[name] = seeds
        rows, summary = _collect_bc_visits(
            model, seeds, deterministic=deterministic,
            deduplicate_collisions=collision_counts is not None,
            collision_state_counts=collision_counts)
        stages.append({"stage": name, "kind": "dual_nn_visitation", **summary})
        return rows

    source = teacher_stage("initial_public_teacher", 10,
                           BC_INITIAL_TEACHER_EPISODES)
    tree_initialization = _initialize_public_trees(model, *source[:3])
    optimizer = torch.optim.Adam(
        model.actors.parameters(),
        lr=REGISTERED_BC_SPEC["initial_bc_learning_rate"])
    generator = torch.Generator().manual_seed(pair_seed + 31)
    initial_accuracy = _bc_epochs(
        model, optimizer, *source[:3], epochs=BC_INITIAL_EPOCHS,
        generator=generator)
    stages.append({"stage": "initial_public_bc", "training_rows": len(source[0]),
                   "epochs": BC_INITIAL_EPOCHS,
                   **initial_fit_config,
                   "public_label_accuracy": initial_accuracy,
                   "actor_role_sha256": role_hashes(model)})

    early_parts = [source]
    for number in range(1, BC_INITIAL_DAGGER_ROUNDS + 1):
        visited = visit_stage(f"initial_nn_dagger_{number}", 10 + number,
                              BC_INITIAL_DAGGER_EPISODES)
        early_parts.append(visited)
        optimizer = torch.optim.Adam(
            model.actors.parameters(),
            lr=REGISTERED_BC_SPEC["initial_bc_learning_rate"])
        obs, mask, labels = (
            torch.cat([part[index] for part in early_parts]) for index in range(3))
        accuracy = _bc_epochs(model, optimizer, obs, mask, labels,
                              epochs=BC_INITIAL_DAGGER_EPOCHS,
                              generator=generator)
        stages.append({"stage": f"initial_nn_dagger_{number}_fit",
                       "training_rows": len(obs),
                       "training_data_sha256": _bc_rows_sha256(
                           (obs, mask, labels)),
                       "epochs": BC_INITIAL_DAGGER_EPOCHS,
                       **initial_fit_config,
                       "public_label_accuracy": accuracy,
                       "actor_role_sha256": role_hashes(model)})

    critic, critic_report = train_collision_critic(
        seed=expected_critic_resets["seed"],
        epochs=REGISTERED_BC_SPEC["collision_critic_epochs"],
        batch_size=REGISTERED_BC_SPEC["collision_critic_batch_size"],
        learning_rate=REGISTERED_BC_SPEC["collision_critic_learning_rate"],
        positive_weight=REGISTERED_BC_SPEC["collision_critic_positive_weight"])
    if (not critic_report.get("validation_passed")
            or critic_report.get("version") !=
            REGISTERED_BC_SPEC["collision_critic_version"]
            or critic_report.get("epochs") !=
            REGISTERED_BC_SPEC["collision_critic_epochs"]
            or critic_report.get("batch_size") !=
            REGISTERED_BC_SPEC["collision_critic_batch_size"]
            or critic_report.get("learning_rate") !=
            REGISTERED_BC_SPEC["collision_critic_learning_rate"]
            or critic_report.get("positive_weight") !=
            REGISTERED_BC_SPEC["collision_critic_positive_weight"]
            or critic_report.get("environment_reset_seed") !=
            expected_critic_resets["seed"]
            or critic_report.get("environment_reset_count") !=
            expected_critic_resets["environment_reset_count"]):
        raise RuntimeError("Warehouse BC real-event collision critic failed validation")
    critic.eval()
    for parameter in critic.parameters():
        parameter.requires_grad_(False)

    risk_teacher = teacher_stage("risk_public_teacher", 20,
                                 BC_RISK_TEACHER_EPISODES)
    risk_visit = visit_stage("risk_nn_visitation", 21,
                             BC_RISK_VISIT_EPISODES)
    risk_fit = _risk_bc_epochs(
        model, critic, (risk_teacher, risk_visit), epochs=BC_RISK_EPOCHS,
        generator_seed=pair_seed + 201,
        margin_weight=0.0, near_increment=0.0,
        low_return_increment=BC_LOW_RETURN_INCREMENT)
    stages.append({"stage": "risk_public_bc_fit", "fit": risk_fit,
                   "actor_role_sha256": role_hashes(model)})

    large_teacher = teacher_stage("large_public_teacher", 30,
                                  BC_LARGE_TEACHER_EPISODES)
    large_visit = visit_stage("large_nn_visitation", 31,
                              BC_LARGE_VISIT_EPISODES)
    large_fit = _risk_bc_epochs(
        model, critic, (large_teacher, large_visit), epochs=BC_RISK_EPOCHS,
        generator_seed=pair_seed + 301,
        margin_weight=BC_MARGIN_WEIGHT, near_increment=BC_NEAR_INCREMENT,
        low_return_increment=BC_LOW_RETURN_INCREMENT)
    stages.append({"stage": "large_public_bc_fit", "fit": large_fit,
                   "actor_role_sha256": role_hashes(model)})

    deep_parts = [teacher_stage("deep_public_teacher", 40,
                                BC_DEEP_TEACHER_EPISODES)]
    for number in range(1, BC_DEEP_DAGGER_ROUNDS + 1):
        counts: dict[bytes, int] = {}
        sampled = visit_stage(f"deep_{number}_sampled", 40 + 2 * number - 1,
                              BC_DEEP_DAGGER_EPISODES // 2,
                              collision_counts=counts)
        argmax = visit_stage(f"deep_{number}_argmax", 40 + 2 * number,
                             BC_DEEP_DAGGER_EPISODES // 2,
                             deterministic=True, collision_counts=counts)
        deep_parts.extend((sampled, argmax))
        fit = _risk_bc_epochs(
            model, critic, deep_parts, epochs=BC_DEEP_EPOCHS,
            generator_seed=pair_seed + 400 + number,
            margin_weight=BC_MARGIN_WEIGHT, near_increment=BC_NEAR_INCREMENT,
            low_return_increment=BC_LOW_RETURN_INCREMENT)
        stages.append({"stage": f"deep_dagger_{number}_fit", "fit": fit,
                       "actor_role_sha256": role_hashes(model)})
    final_fit = _risk_bc_epochs(
        model, critic, deep_parts, epochs=BC_DEEP_EPOCHS,
        generator_seed=pair_seed + 499,
        margin_weight=BC_MARGIN_WEIGHT, near_increment=BC_NEAR_INCREMENT,
        low_return_increment=BC_LOW_RETURN_INCREMENT)
    stages.append({"stage": "final_standard_margin_calibration", "fit": final_fit,
                   "actor_role_sha256": role_hashes(model)})

    all_seeds = [seed for block in all_seed_blocks.values() for seed in block]
    if (all_seed_blocks != expected_seed_blocks
            or len(set(all_seeds)) != len(all_seeds)):
        raise RuntimeError("Warehouse BC seed phases overlap or drifted")
    obs, mask, labels = (
        torch.cat([part[index] for part in deep_parts]) for index in range(3))
    return {
        "source": "public_observation_only_offline_bc_and_dual_nn_visits",
        "bc_recipe_version": BC_RECIPE_VERSION,
        "bc_recipe_sha256": REGISTERED_BC_SPEC_SHA256,
        "risk_dual_config_sha256": REGISTERED_RISK_DUAL_CONFIG_SHA256,
        "source_closure_sha256": spec["source_closure_sha256"],
        "observation_version": OBSERVATION_VERSION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "initial_role_sha256": initial_hashes,
        "pretrained_role_sha256": role_hashes(model),
        "tree_initialization": tree_initialization,
        "collision_critic": critic_report,
        "bc_data_seed_blocks": all_seed_blocks,
        "bc_data_seed_schedule_sha256": _digest_json(all_seed_blocks),
        "bc_critic_reset_schedule": expected_critic_resets,
        "bc_critic_reset_schedule_sha256": _digest_json(expected_critic_resets),
        "bc_stages": stages,
        "final_bc_training_reliability": _bc_reliability(model, obs, mask, labels),
        "training_dependencies": {
            "torch": torch.__version__, "numpy": np.__version__,
            "sklearn": tree_initialization[AGENTS[0]]["sklearn_version"]},
        "teacher_uses_hidden_goal_or_route_state": False,
        "offline_critic_used_at_execution": False,
    }


def pretrain_pair(model: WarehouseV6Actor, *, pair_seed: int, spec: Mapping) -> dict:
    """Train one shared dual-NN BC start for paired raw and regularized PPO."""
    if spec.get("smoke") is True:
        return _pretrain_pair_smoke(model, pair_seed=pair_seed, spec=spec)
    return _pretrain_pair_production(model, pair_seed=pair_seed, spec=spec)


def collect_episode(model: WarehouseV6Actor, *, seed: int,
                    generator: torch.Generator, max_steps: int,
                    episode_index: int = 0):
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
            cost_value = model.cost_critic(t_obs.flatten(start_dim=1))[0, 0].sigmoid()
        commands = {role: ACTIONS[int(action[index])]
                    for index, role in enumerate(AGENTS)}
        previous_shutdowns = env.state.shutdown_count
        episode_id, frame = int(env.state.episode_id), int(env.state.frame)
        _, reward, _, _, info = env.step(commands)
        if info["proposed_actions"] != commands or info["action_override_agents"]:
            raise RuntimeError("Warehouse v6 formal NN action changed before physics")
        records.append({"obs": obs, "mask": mask, "action": action.numpy(),
                        "logp": logp.numpy(), "reward_value": reward_value.numpy().copy(),
                        "cost_value": float(cost_value),
                        "reward": np.asarray([reward[role] for role in AGENTS], dtype=np.float32),
                        "shutdown_cost": float(env.state.shutdown_count > previous_shutdowns),
                        "seed": seed, "frame": frame,
                        "episode_id": episode_id,
                        "episode_key": f"{seed}:{episode_index}"})
    complete = bool(env.state.terminated or env.state.truncated)
    if complete:
        reward_bootstrap = np.zeros(len(AGENTS), dtype=np.float32)
        cost_bootstrap = 0.0
    else:
        with torch.no_grad():
            final_obs = torch.from_numpy(encode_public_observation(env.state)).flatten()[None]
            reward_bootstrap = model.reward_critic(final_obs)[0].numpy().copy()
            cost_bootstrap = float(model.cost_critic(final_obs)[0, 0].sigmoid())
    summary = {"seed": seed, "episode_index": episode_index,
               "episode_key": f"{seed}:{episode_index}",
               "joint_steps": len(records),
               "actor_decisions": len(records) * len(AGENTS),
               "post_policy_interventions": 0,
               "deliveries": env.state.total_deliveries,
               "collisions": env.state.robot_collision_events,
               "shutdown_episode": bool(env.state.shutdown_count),
               "complete": complete}
    return {"records": records, "reward_bootstrap": reward_bootstrap,
            "cost_bootstrap": cost_bootstrap, "complete": complete,
            "shutdown_episode": summary["shutdown_episode"]}, summary


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


def _ppo_rows(episodes: Sequence[Mapping]) -> dict:
    rows = []
    complete_outcomes = {}
    for identifier, episode in enumerate(episodes):
        if type(episode.get("complete")) is not bool:
            raise ValueError("Warehouse cost PPO requires an explicit complete flag")
        records = episode["records"]
        if not records:
            raise RuntimeError("Warehouse v6 PPO episode has no transitions")
        if episode["complete"]:
            costs = [float(item["shutdown_cost"]) for item in records]
            values = [float(item["cost_value"]) for item in records]
            cost_advantages, _ = shutdown_cost_gae(
                costs, values, complete=True)
            outcome = bool(sum(costs))
            if episode.get("shutdown_episode") not in (None, outcome):
                raise RuntimeError("Warehouse episode shutdown outcome disagrees with events")
            complete_outcomes[identifier] = outcome
        else:
            if any(float(item["shutdown_cost"]) for item in records):
                raise RuntimeError("Warehouse incomplete episode has a shutdown event")
            cost_advantages = torch.zeros(len(records))
        reward_adv = np.zeros(len(AGENTS), dtype=np.float32)
        next_reward = episode["reward_bootstrap"]
        enriched = []
        for index in range(len(records) - 1, -1, -1):
            item = records[index]
            reward_delta = item["reward"] + 0.99 * next_reward - item["reward_value"]
            reward_adv = reward_delta + 0.99 * 0.95 * reward_adv
            enriched.append((item, reward_adv.copy(), reward_adv + item["reward_value"],
                             float(cost_advantages[index]), identifier))
            next_reward = item["reward_value"]
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
        "episode_id": torch.tensor([row[4] for row in rows], dtype=torch.long),
        "complete_episode_outcomes": complete_outcomes,
    }


def ppo_update(model: WarehouseV6Actor, optimizer: torch.optim.Optimizer,
               episodes: Sequence[Mapping], programs: Mapping[str, WarehouseV6Program | None] | None,
               *, progress: float, cost_multiplier: float, epochs: int = 4,
               program_validation_rows: Mapping[str, Sequence[Mapping]] | None = None) -> dict:
    data = _ppo_rows(episodes)
    obs, masks, actions, old_logp = (data[key] for key in
                                     ("obs", "mask", "actions", "old_logp"))
    encoded = policy_features_torch(obs, masks)
    reward_adv = data["reward_adv"]
    reward_adv = ((reward_adv - reward_adv.mean(0, keepdim=True)) /
                  reward_adv.std(0, keepdim=True, unbiased=False).clamp_min(1e-6))
    cost_adv = data["cost_adv"]
    complete_outcomes = data["complete_episode_outcomes"]
    cost_event_rows = sum(complete_outcomes.values())
    complete_values = cost_adv[torch.isin(
        data["episode_id"], torch.tensor(list(complete_outcomes), dtype=torch.long))]
    cost_adv_std_raw = (float(complete_values.std(unbiased=False))
                        if len(complete_values) else 0.0)
    coefficient = _program_coefficient(progress) if programs is not None else 0.0
    counts = {role: 0 for role in AGENTS}
    gradient_norms = {role: 0.0 for role in AGENTS}
    exposure = {role: 0 for role in AGENTS}
    zero_coverage = {role: 0 for role in AGENTS}
    no_legal = {role: 0 for role in AGENTS}
    program_kl = {role: 0.0 for role in AGENTS}
    feedback_disabled_epochs = {role: 0 for role in AGENTS}
    current_validation = {role: [] for role in AGENTS}
    split_gradient_norm = {role: 0.0 for role in AGENTS}
    losses = []
    for _ in range(epochs):
        epoch_programs = {role: None for role in AGENTS}
        if coefficient > 0 and programs is not None:
            # The prior checkpoint's certificate expires after any optimizer
            # step. Recheck the current complete Actor on fixed, independent
            # validation states before *each* epoch's program gradient.
            current_actor = FrozenWarehouseV6Actor.from_model(model)
            for role in AGENTS:
                program = programs.get(role)
                fixed = (program_validation_rows or {}).get(role, ())
                if program is None or not fixed:
                    feedback_disabled_epochs[role] += 1
                    current_validation[role].append({
                        "passed": False, "reason": "missing_program_or_fixed_validation"})
                    continue
                rows = _resample_program_rows(current_actor, role, fixed)
                metric = current_actor_fidelity(program, current_actor, rows,
                                                purpose="feedback")
                metric["fixed_validation_rows_sha256"] = canonical_sha256(fixed)
                current_validation[role].append(metric)
                if metric["passed"]:
                    epoch_programs[role] = program
                else:
                    feedback_disabled_epochs[role] += 1
        logits = model.outputs_encoded(encoded, masks)
        policy = Categorical(logits=logits)
        new_logp = policy.log_prob(actions)
        ratio = torch.exp(new_logp - old_logp)
        reward_surrogate = torch.minimum(
            ratio * reward_adv, ratio.clamp(0.8, 1.2) * reward_adv)
        # The shutdown cost is an observed episode event, not a hand-written
        # action-risk matrix or a run-time safety controller.
        cost_surrogate, cost_diagnostics = joint_shutdown_surrogate(
            new_logp, old_logp, cost_adv, data["episode_id"],
            tuple(complete_outcomes))
        reward_value = model.reward_critic(obs.flatten(start_dim=1))
        cost_logits = model.cost_critic(obs.flatten(start_dim=1)).squeeze(-1)
        cost_critic_loss = episode_balanced_cost_critic_loss(
            cost_logits, data["episode_id"], complete_outcomes)
        loss = (-reward_surrogate.mean() + cost_multiplier * cost_surrogate
                + 0.5 * nn.functional.mse_loss(reward_value, data["reward_target"])
                + 0.25 * cost_critic_loss
                - 0.005 * policy.entropy().mean()
                + 5e-4 * torch.stack([
                    model.actors[role].bounds().mean() for role in AGENTS]).mean())
        if programs is not None:
            for index, role in enumerate(AGENTS):
                penalty, count, kl = _program_feedback(
                    logits[:, index], obs[:, index], masks[:, index], actions[:, index],
                    epoch_programs[role], coefficient=coefficient)
                if count:
                    grads = torch.autograd.grad(
                        penalty, tuple(parameter for parameter in
                                       model.actors[role].parameters()
                                       if parameter.requires_grad),
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
        for role in AGENTS:
            split = model.actors[role].tree.split
            split_gradient_norm[role] += float(sum(
                parameter.grad.detach().square().sum().item()
                for parameter in (split.weight, split.bias) if parameter.grad is not None) ** 0.5)
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        losses.append(float(loss.detach()))
    return {"loss": float(np.mean(losses)), "cost_multiplier": cost_multiplier,
            "cost_event_rows": cost_event_rows,
            "completed_episodes_for_cost": len(complete_outcomes),
            "cost_advantage_std_raw": cost_adv_std_raw,
            "cost_actor_surrogate": float(cost_surrogate.detach()),
            "cost_joint_ratio_mean": cost_diagnostics["joint_ratio_mean"],
            "cost_critic_loss": float(cost_critic_loss.detach()),
            "split_gradient_norm": split_gradient_norm,
            "program_coefficient": coefficient, "feedback_rows": counts,
            "feedback_gradient_norm": gradient_norms, "program_kl": program_kl,
            "feedback_disabled_epochs": feedback_disabled_epochs,
            "current_actor_validation": current_validation,
            "feedback_window_actor_rows": exposure,
            "zero_action_coverage_rows": zero_coverage,
            "physical_no_legal_action_rows": no_legal,
            "batch_shutdown_episode_rate": (sum(complete_outcomes.values()) /
                                             len(complete_outcomes)
                                             if complete_outcomes else None),
            "cost_critic_target_mean": (sum(complete_outcomes.values()) /
                                        len(complete_outcomes)
                                        if complete_outcomes else None)}


def evaluate(model: WarehouseV6Actor | FrozenWarehouseV6Actor,
             seeds: Sequence[int], *, action_seed_offset: int = 0,
             trace: bool = False) -> dict:
    frozen = (FrozenWarehouseV6Actor.from_model(model)
              if isinstance(model, WarehouseV6Actor) else model)
    if trace and (len(frozen.artifact_sha256) != 64 or
                  any(char not in "0123456789abcdef"
                      for char in frozen.artifact_sha256)):
        raise ValueError("Warehouse quality trace requires a frozen artifact SHA")
    episodes = []
    for seed in seeds:
        env = _environment(training_reward=False)
        env.reset(seed=int(seed))
        _set_dual_nn_mode(env)
        action_seed = int(seed) + action_seed_offset
        encoded_actions = bytearray()
        first_pre_sha, last_post_sha = None, None
        if trace:
            header = {
                "schema": "warehouse-quality-trace.v1",
                "actor_sha256": frozen.artifact_sha256,
                "scenario_seed": int(seed),
                "action_seed": action_seed,
                "episode_id": int(env.state.episode_id),
            }
            chain = bytes.fromhex(_digest_json(header))
        while not (env.state.terminated or env.state.truncated):
            if trace:
                frame = int(env.state.frame)
                pre_sha = canonical_sha256(public_pre_action_snapshot(env.state))
                if first_pre_sha is None:
                    first_pre_sha = pre_sha
            actions, _ = frozen.act(env.state, base_seed=action_seed)
            _, _, _, _, info = env.step(actions)
            if info["proposed_actions"] != actions or info["action_override_agents"]:
                raise RuntimeError("Warehouse v6 evaluator rewrote a frozen NN action")
            if trace:
                a1, a2 = (ACTIONS.index(actions[role]) for role in AGENTS)
                encoded_actions.append(len(ACTIONS) * a1 + a2)
                last_post_sha = canonical_sha256(public_pre_action_snapshot(env.state))
                frame_bytes = json.dumps(
                    [frame, pre_sha, a1, a2, last_post_sha],
                    sort_keys=True, separators=(",", ":"), allow_nan=False,
                ).encode("utf-8")
                chain = sha256(chain + frame_bytes).digest()
        row = {"seed": int(seed), "deliveries": env.state.total_deliveries,
               "collisions": env.state.robot_collision_events,
               "shutdown_episode": bool(env.state.shutdown_count)}
        if trace:
            if not encoded_actions or first_pre_sha is None or last_post_sha is None:
                raise RuntimeError("Warehouse quality trace has no executed NN frame")
            row["quality_trace"] = {
                **header,
                "actor_role_sha256": dict(frozen.role_sha256),
                "frame_count": len(encoded_actions),
                "action_encoding": "base64-u8-a1-times-5-plus-a2",
                "actions_b64": base64.b64encode(bytes(encoded_actions)).decode("ascii"),
                "initial_pre_state_sha256": first_pre_sha,
                "final_post_state_sha256": last_post_sha,
                "chain_sha256": chain.hex(),
            }
        episodes.append(row)
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
             "episode": str(record["episode_key"]),
             "base_seed": int(record["seed"]),
             "episode_id": int(record["episode_id"]),
             "frame": int(record["frame"]),
             "historical_sampled_action": ACTIONS[int(record["action"][role_index])],
             "sampled_action": ACTIONS[int(record["action"][role_index])]}
            for episode in episodes for record in episode["records"]]


def _resample_program_rows(actor: FrozenWarehouseV6Actor, role: str,
                           rows: Sequence[Mapping]) -> list[dict]:
    """Sample the *current* frozen NN on fixed public pre-action states.

    Buffer actions came from earlier behavior policies and cannot establish
    current-Actor fidelity. The same per-row random stream is reused when a
    later PPO epoch rechecks that Actor against these held validation states.
    """
    output = []
    for begin in range(0, len(rows), 256):
        chunk = rows[begin:begin + 256]
        obs = np.stack([row["obs"] for row in chunk])
        masks = np.stack([row["mask"] for row in chunk])
        encoded = np.concatenate((obs, masks), axis=-1)
        logits = np.where(masks > 0.5, actor.roles[role].logits_encoded(encoded), -1e9)
        shifted = logits - logits.max(-1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(-1, keepdims=True)
        for row, p in zip(chunk, probabilities, strict=True):
            seed = independent_agent_seed(
                base_seed=int(row["base_seed"]), episode_id=int(row["episode_id"]),
                frame=int(row["frame"]), agent_id=role)
            action = ACTIONS[int(np.random.default_rng(seed).choice(len(ACTIONS), p=p))]
            output.append({**row, "sampled_action": action,
                           "current_actor_role_sha256": actor.role_sha256[role],
                           "validation_rng_seed": seed})
    return output


def _fit_training_programs(model: WarehouseV6Actor,
                           buffers: Mapping[str, deque], *,
                           method: str = AXIS_TREE_METHOD) -> tuple[dict, dict, dict]:
    if method not in (AXIS_TREE_METHOD, ACTOR_SPARSE_TOPOLOGY_METHOD):
        raise ValueError("unregistered Warehouse online program method")
    actor = FrozenWarehouseV6Actor.from_model(model)
    programs, reports, held_validation = {}, {}, {}
    split_names = tuple(FINAL_PROGRAM_SPLIT_COUNTS)
    if tuple(FINAL_PROGRAM_SPLIT_COUNTS.values()) != PROGRAM_SPLIT_EPISODES:
        raise RuntimeError("Warehouse online program split counts drifted")
    for role in AGENTS:
        rows = list(buffers[role])
        # One training scenario block can reset the same seed several times
        # after early termination. Select *distinct scenario seeds*, keeping
        # every reset of one seed in the same purpose. The most recent 124
        # complete blocks fit inside the 20k-row buffer at every extraction.
        scenario_seeds = list(dict.fromkeys(int(row["base_seed"]) for row in rows))
        if len(scenario_seeds) < PROGRAM_MIN_EPISODES:
            programs[role] = None
            reports[role] = {"error": "insufficient distinct training scenario episodes",
                             "program_fit_method": method,
                             "episodes": len(scenario_seeds)}
            held_validation[role] = []
            continue
        selected = scenario_seeds[-PROGRAM_MIN_EPISODES:]
        groups, cursor = {}, 0
        for name, count in zip(split_names, PROGRAM_SPLIT_EPISODES, strict=True):
            groups[name] = set(selected[cursor:cursor + count])
            cursor += count
        if cursor != len(selected) or len(set().union(*groups.values())) != cursor:
            raise RuntimeError("Warehouse online program cohort is not scene-disjoint")
        parts = {name: _resample_program_rows(actor, role, [
            row for row in rows if int(row["base_seed"]) in groups[name]])
            for name in split_names}
        held_validation[role] = parts["validation"]
        provenance = {
            "training_split_scenario_seeds": {
                name: [seed for seed in selected if seed in group]
                for name, group in groups.items()},
            "training_split_episode_ids": {
                name: list(dict.fromkeys(str(row["episode"]) for row in parts[name]))
                for name in split_names},
            "training_split_episode_counts": {
                name: len(set(row["episode"] for row in parts[name]))
                for name in split_names},
        }
        try:
            if method == AXIS_TREE_METHOD:
                # Preserve the historical smoke and direct unit-test adapter.
                program = fit_program(role, actor, *(parts[name] for name in split_names))
            else:
                program = fit_program(role, actor, *(parts[name] for name in split_names),
                                      method=method)
            programs[role] = program
            reports[role] = {"program_sha256": program.sha256(),
                             "program_fit_method": method,
                             "actor_role_sha256": actor.role_sha256[role],
                             **provenance,
                             "current_actor_resampled": True,
                             "fixed_validation_rows_sha256": canonical_sha256(
                                 parts["validation"]),
                             "validation": program.report["validation"]}
        except (ValueError, RuntimeError) as exc:
            programs[role] = None
            reports[role] = {"error": f"{type(exc).__name__}: {exc}",
                             "program_fit_method": method,
                             **provenance}
    return programs, reports, held_validation


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
            pre_state = public_pre_action_snapshot(env.state)
            rebuilt = rebuild_public_pre_action_state(pre_state)
            if (not np.array_equal(obs, encode_public_observation(rebuilt))
                    or not np.array_equal(masks, action_masks(rebuilt))):
                raise RuntimeError("Warehouse audit public snapshot cannot rebuild NN inputs")
            pre_sha = canonical_sha256(pre_state)
            frame, episode_id = int(env.state.frame), int(env.state.episode_id)
            actions, distributions = actor.act(env.state, base_seed=int(seed))
            for index, role in enumerate(AGENTS):
                distribution = distributions[role]
                rows[role].append({"obs": obs[index].copy(),
                                   "mask": masks[index].copy(),
                                   "episode": str(seed),
                                   "episode_id": episode_id, "frame": frame,
                                   "base_seed": int(seed), "role": role,
                                   "sampled_action": actions[role],
                                   "nn_sampled_action": actions[role],
                                   "nn_argmax_action": ACTIONS[int(np.argmax(
                                       distribution.probabilities))],
                                   "nn_probabilities": list(distribution.probabilities),
                                   "rng_seed": independent_agent_seed(
                                       base_seed=int(seed), episode_id=episode_id,
                                       frame=frame, agent_id=role),
                                   "actor_sha256": actor.artifact_sha256,
                                   "actor_role_sha256": actor.role_sha256[role],
                                   "pre_state_public": pre_state,
                                   "pre_state_public_sha256": pre_sha})
            _, _, _, _, info = env.step(actions)
            if info["proposed_actions"] != actions or info["action_override_agents"]:
                raise RuntimeError("Warehouse v6 audit rollout rewrote an NN action")
            post_state = public_pre_action_snapshot(env.state)
            post_sha = canonical_sha256(post_state)
            for role in AGENTS:
                rows[role][-1]["post_state_public"] = post_state
                rows[role][-1]["post_state_public_sha256"] = post_sha
    return rows


def audit_programs(actor: FrozenWarehouseV6Actor,
                   programs: Mapping[str, WarehouseV6Program],
                   seeds: Sequence[int], *,
                   trace_dir: str | Path | None = None) -> dict[str, dict]:
    if set(programs) != set(AGENTS):
        raise ValueError("Warehouse v6 audit requires both role programs")
    samples = collect_program_audit_rows(actor, seeds)
    result = {}
    for role in AGENTS:
        if programs[role].actor_role_sha256 != actor.role_sha256[role]:
            raise RuntimeError("Warehouse audit program Actor hash mismatch")
        program_sha = programs[role].sha256()
        for row in samples[role]:
            row["program_sha256"] = program_sha
        result[role] = audit_frozen_program(programs[role], actor, samples[role])
        if trace_dir is not None:
            target_root = Path(trace_dir).resolve()
            target_root.mkdir(parents=True, exist_ok=True)
            data = b"".join((json.dumps({
                **row, "obs": row["obs"].tolist(), "mask": row["mask"].tolist()},
                sort_keys=True, separators=(",", ":"), allow_nan=False
            ) + "\n").encode() for row in samples[role])
            digest = sha256(data).hexdigest()
            pointer = target_root / f"{role}.sha256"
            if pointer.is_file() and pointer.read_text().strip() != digest:
                raise RuntimeError("Warehouse immutable audit trace already has a different hash")
            target = target_root / digest / f"{role}.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file():
                if target.read_bytes() != data:
                    raise RuntimeError("Warehouse content-addressed audit trace changed")
            else:
                with target.open("xb") as handle:
                    handle.write(data)
            if not pointer.exists():
                with pointer.open("x") as handle:
                    handle.write(digest + "\n")
            result[role].update(trace_path=str(target), trace_sha256=digest,
                                trace_row_count=len(samples[role]))
    return result


def _write_final_fit_trace(rows: Mapping[str, Sequence[Mapping]], *,
                           split: str, seeds: Sequence[int],
                           trace_root: Path, actor_sha256: str) -> dict:
    """Save an immutable joint NN trajectory before any program is fitted."""
    if split not in ("fit", "calibration", "eligibility", "validation"):
        raise ValueError("invalid Warehouse final program purpose")
    first, second = rows[AGENTS[0]], rows[AGENTS[1]]
    if not first or len(first) != len(second):
        raise RuntimeError("Warehouse final program role trajectories are misaligned")
    lines = []
    seen_seeds, prior_seed, prior_frame = [], None, None
    for r1, r2 in zip(first, second, strict=True):
        for key in ("base_seed", "episode_id", "frame",
                    "pre_state_public_sha256", "post_state_public_sha256"):
            if r1[key] != r2[key]:
                raise RuntimeError("Warehouse final program role frames differ")
        if (r1["pre_state_public"] != r2["pre_state_public"]
                or r1["post_state_public"] != r2["post_state_public"]
                or r1["actor_sha256"] != actor_sha256
                or r2["actor_sha256"] != actor_sha256):
            raise RuntimeError("Warehouse final program pre/post state or Actor differs")
        seed, frame = int(r1["base_seed"]), int(r1["frame"])
        if seed != prior_seed:
            if frame != 0 or seed in seen_seeds:
                raise RuntimeError("Warehouse final program episode order changed")
            seen_seeds.append(seed)
        elif frame != prior_frame + 1:
            raise RuntimeError("Warehouse final program has a missing frame")
        prior_seed, prior_frame = seed, frame
        role_rows = {}
        for role, row in zip(AGENTS, (r1, r2), strict=True):
            role_rows[role] = {
                "obs": np.asarray(row["obs"], dtype=np.float32).tolist(),
                "mask": np.asarray(row["mask"], dtype=np.float32).tolist(),
                "sampled_action": row["sampled_action"],
                "nn_probabilities": list(row["nn_probabilities"]),
                "rng_seed": int(row["rng_seed"]),
                "actor_role_sha256": row["actor_role_sha256"],
            }
        lines.append(json.dumps({
            "base_seed": seed, "episode_id": int(r1["episode_id"]),
            "frame": frame,
            "pre_state_public": r1["pre_state_public"],
            "pre_state_public_sha256": r1["pre_state_public_sha256"],
            "post_state_public": r1["post_state_public"],
            "post_state_public_sha256": r1["post_state_public_sha256"],
            "actor_sha256": actor_sha256, "roles": role_rows,
        }, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    if seen_seeds != list(seeds):
        raise RuntimeError("Warehouse final program trace seed schedule differs")
    data = "".join(lines).encode()
    digest = sha256(data).hexdigest()
    trace_root.mkdir(parents=True, exist_ok=True)
    pointer = trace_root / f"{split}.sha256"
    if pointer.is_file() and pointer.read_text().strip() != digest:
        raise RuntimeError("Warehouse final program split trace is already bound")
    target = trace_root / digest / f"{split}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        if target.read_bytes() != data:
            raise RuntimeError("Warehouse final program content-addressed trace changed")
    else:
        with target.open("xb") as handle:
            handle.write(data)
    if not pointer.exists():
        with pointer.open("x") as handle:
            handle.write(digest + "\n")
    return {"path": str(target.resolve()), "sha256": digest,
            "row_count": len(lines), "seeds": list(seeds)}


def _fresh_final_programs(model: WarehouseV6Actor,
                          spec: Mapping, histories: Sequence[Mapping], *,
                          directory: Path, actor_artifact: Mapping
                          ) -> tuple[dict, dict]:
    """Refit after checkpoint selection on new, development-only NN episodes.

    Online validation repeatedly gates PPO gradients and is consequently
    adaptive. It is never reused as the frozen program's validation receipt.
    """
    actor = FrozenWarehouseV6Actor.load(
        actor_artifact["path"], expected_sha256=actor_artifact["sha256"],
        expected_role_sha256=role_hashes(model))
    if spec.get("program_fit_method") not in PROGRAM_FIT_METHOD_CANDIDATES:
        raise RuntimeError("Warehouse frozen program method is unsupported")
    from experiments.warehouse.protocol import final_program_fit_seed_schedule
    parts = spec["final_program_fit_seed_schedule"]
    if (parts != final_program_fit_seed_schedule(spec["release_id"], spec["pair_seed"])
            or _digest_json(parts) != spec["final_program_fit_seed_schedule_sha256"]
            or tuple(len(parts[name]) for name in
                ("fit", "calibration", "eligibility", "validation"))
                != PROGRAM_SPLIT_EPISODES):
        raise RuntimeError("Warehouse frozen program fit schedule drifted")
    seeds = [seed for name in ("fit", "calibration", "eligibility", "validation")
             for seed in parts[name]]
    excluded = {int(row["seed"]) for row in histories} | set(spec["checkpoint_seeds"])
    if excluded & set(seeds):
        raise RuntimeError("Warehouse final program episodes overlap training or selection")
    rows = collect_program_audit_rows(actor, seeds)
    assigned = {name: set(values) for name, values in parts.items()}
    split_rows = {
        name: {role: [row for row in rows[role]
                      if int(row["base_seed"]) in assigned[name]]
               for role in AGENTS}
        for name in ("fit", "calibration", "eligibility", "validation")
    }
    traces = {name: _write_final_fit_trace(
        split_rows[name], split=name, seeds=parts[name],
        trace_root=directory / "final_fit_traces",
        actor_sha256=actor.artifact_sha256)
        for name in ("fit", "calibration", "eligibility", "validation")}
    programs, reports = {}, {}
    for role in AGENTS:
        split = {name: split_rows[name][role] for name in split_rows}
        try:
            program = fit_program(role, actor, *(split[name] for name in
                ("fit", "calibration", "eligibility", "validation")),
                method=spec["program_fit_method"])
            if program.report.get("fit_method") != spec["program_fit_method"]:
                raise RuntimeError("Warehouse frozen program used a different method")
            programs[role] = program
            reports[role] = {
                "status": "frozen_actor_fresh_four_part_fit",
                "program_fit_method": spec["program_fit_method"],
                "actor_role_sha256": actor.role_sha256[role],
                "actor_artifact_sha256": actor.artifact_sha256,
                "program_sha256": program.sha256(),
                "release_id": spec["release_id"],
                "final_fit_traces": traces,
                "split_seeds": parts,
                "split_seed_schedule_sha256": spec[
                    "final_program_fit_seed_schedule_sha256"],
                "validation": program.report["validation"],
            }
        except (ValueError, RuntimeError) as exc:
            reports[role] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                             "program_fit_method": spec["program_fit_method"],
                             "release_id": spec["release_id"],
                             "actor_artifact_sha256": actor.artifact_sha256,
                             "final_fit_traces": traces,
                             "split_seeds": parts,
                             "split_seed_schedule_sha256": spec[
                                 "final_program_fit_seed_schedule_sha256"]}
    return programs, reports


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


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash every parameter and buffer independently of torch serialization."""
    digest = sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise RuntimeError("Warehouse checkpoint contains a non-tensor state")
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


_RESUME_PROGRESS_FIELDS = (
    "used_steps", "episode_number", "histories", "updates", "development",
    "next_eval", "next_extraction_index", "last_extraction_step",
    "risk_dual_snapshot", "extractions",
)


def _resume_progress_sha256(checkpoint: Mapping) -> str:
    return _digest_json({name: checkpoint[name] for name in
                         _RESUME_PROGRESS_FIELDS})


def _verify_episode_progress(histories: Sequence[Mapping], used_steps: int,
                             episode_number: int) -> None:
    """A resumed trajectory must cover exactly the reported joint steps."""
    if (type(used_steps) is not int or used_steps < 0
            or type(episode_number) is not int
            or episode_number != len(histories)):
        raise RuntimeError("Warehouse checkpoint episode count or steps changed")
    cursor = 0
    for index, row in enumerate(histories):
        steps = row.get("joint_steps")
        if (type(steps) is not int or steps <= 0
                or row.get("start_joint_step") != cursor
                or row.get("episode_index") != index):
            raise RuntimeError("Warehouse checkpoint episode trajectory changed")
        cursor += steps
    if cursor != used_steps:
        raise RuntimeError("Warehouse checkpoint episode steps changed")


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
    spec_hash = _training_config_sha256(spec)
    run_spec_sha256 = _scientific_run_spec_sha256(spec)
    used_steps, episode_number = 0, 0
    histories, updates, development = [], [], []
    buffers = {role: deque(maxlen=PROGRAM_BUFFER_STATES) for role in AGENTS}
    programs = {role: None for role in AGENTS} if variant == "regularized" else None
    program_validation_rows = {role: [] for role in AGENTS}
    extractions = {role: [] for role in AGENTS}
    extraction_steps = tuple(spec.get("online_program_extraction_steps", ()))
    next_extraction_index = 0
    last_extraction_step = 0
    best = None
    risk_dual = ShutdownRiskDual(RiskDualConfig())
    cost_multiplier = risk_dual.multiplier
    interval = max(1, spec["joint_steps"] // 10)
    next_eval = interval
    if checkpoint.is_file():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (saved.get("training_config_sha256") != spec_hash
                or saved.get("source_closure_sha256") != spec["source_closure_sha256"]
                or saved.get("variant") != variant
                or saved.get("initial_role_sha256") != initial_role_hashes
                or saved.get("model_version") != MODEL_VERSION
                or saved.get("feature_schema_sha256") != FEATURE_SCHEMA_SHA256
                or (not spec.get("smoke") and (
                    saved.get("run_spec_sha256") != run_spec_sha256
                    or saved.get("pair_seed") != spec["pair_seed"]
                    or saved.get("joint_steps") != spec["joint_steps"]
                    or saved.get("program_fit_method") != spec["program_fit_method"]
                    or saved.get("bc_recipe_sha256") != spec["bc_recipe_sha256"]
                    or saved.get("bc_data_seed_schedule_sha256") !=
                    spec["bc_data_seed_schedule_sha256"]
                    or saved.get("risk_dual_config_sha256") !=
                    spec["risk_dual_config_sha256"]))):
            raise RuntimeError("Warehouse v6 checkpoint identity or Actor start mismatch")
        if saved.get("complete"):
            if not spec.get("smoke"):
                selected = saved.get("selected_state_dict")
                report = saved.get("report")
                if (not isinstance(selected, Mapping)
                        or not isinstance(report, Mapping)
                        or saved.get("selected_state_sha256") !=
                        _state_dict_sha256(selected)
                        or saved.get("report_sha256") != _digest_json(report)
                        or saved.get("selected_programs_sha256") !=
                        _digest_json(saved.get("selected_programs"))
                        or report.get("run_spec_sha256") != run_spec_sha256
                        or report.get("training_config_sha256") != spec_hash
                        or report.get("source_closure_sha256") !=
                        spec["source_closure_sha256"]
                        or report.get("pair_seed") != spec["pair_seed"]
                        or report.get("variant") != variant
                        or report.get("joint_steps") != spec["joint_steps"]
                        or report.get("actor_decisions") !=
                        2 * spec["joint_steps"]
                        or report.get("program_fit_method") !=
                        spec["program_fit_method"]
                        or report.get("bc_recipe_sha256") !=
                        spec["bc_recipe_sha256"]
                        or report.get("bc_data_seed_schedule_sha256") !=
                        spec["bc_data_seed_schedule_sha256"]
                        or report.get("risk_dual_config_sha256") !=
                        spec["risk_dual_config_sha256"]
                        or report.get("actual_episode_resets_sha256") !=
                        _digest_json(report.get("actual_episode_resets"))):
                    raise RuntimeError("Warehouse completed checkpoint evidence changed")
                _verify_episode_progress(report.get("training_episodes", ()),
                                         report["joint_steps"],
                                         len(report.get("training_episodes", ())))
                if (not report.get("updates")
                        or report["updates"][-1].get("joint_steps") !=
                        spec["joint_steps"]):
                    raise RuntimeError("Warehouse completed checkpoint updates changed")
            model.load_state_dict(saved["selected_state_dict"])
            if (not spec.get("smoke") and
                    report["selected_role_sha256"] != role_hashes(model)):
                raise RuntimeError("Warehouse completed checkpoint Actor hash changed")
            restored = {role: WarehouseV6Program.from_dict(
                payload, actor_role_sha256=role_hashes(model)[role])
                for role, payload in saved.get("selected_programs", {}).items()}
            if (not spec.get("smoke") and variant == "regularized"
                    and any(program.report.get("fit_method") !=
                            spec["program_fit_method"]
                            for program in restored.values())):
                raise RuntimeError("Warehouse resumed selected program method drifted")
            return model, restored, saved["report"]
        if not spec.get("smoke"):
            state = saved.get("state_dict")
            best_saved = saved.get("best")
            if (not isinstance(state, Mapping)
                    or saved.get("state_sha256") != _state_dict_sha256(state)
                    or saved.get("resume_progress_sha256") !=
                    _resume_progress_sha256(saved)
                    or saved.get("best_state_sha256") !=
                    (_state_dict_sha256(best_saved["state_dict"])
                     if best_saved is not None else None)
                    or type(saved.get("used_steps")) is not int
                    or not 0 <= saved["used_steps"] <= spec["joint_steps"]
                    or type(saved.get("next_extraction_index")) is not int
                    or not 0 <= saved["next_extraction_index"] <=
                    len(extraction_steps)):
                raise RuntimeError("Warehouse partial checkpoint evidence changed")
            _verify_episode_progress(saved.get("histories", ()),
                                     saved["used_steps"], saved.get("episode_number"))
            if (not saved.get("updates") or
                    saved["updates"][-1].get("joint_steps") !=
                    saved["used_steps"]):
                raise RuntimeError("Warehouse partial checkpoint updates changed")
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["generator_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        used_steps, episode_number = saved["used_steps"], saved["episode_number"]
        histories, updates, development = (saved[key] for key in
                                            ("histories", "updates", "development"))
        best, next_eval = saved["best"], saved["next_eval"]
        risk_dual = ShutdownRiskDual.restore(saved["risk_dual_snapshot"],
                                             RiskDualConfig())
        cost_multiplier = risk_dual.multiplier
        if programs is not None:
            for role in AGENTS:
                buffers[role].extend(saved["buffers"][role])
                payload = saved["programs"][role]
                programs[role] = (WarehouseV6Program.from_dict(
                    payload, actor_role_sha256=payload["actor_role_sha256"])
                    if payload is not None else None)
                if (not spec.get("smoke") and programs[role] is not None
                        and programs[role].report.get("fit_method") !=
                        spec["program_fit_method"]):
                    raise RuntimeError("Warehouse resumed online program method drifted")
            extractions = saved["extractions"]
            program_validation_rows = saved["program_validation_rows"]
        next_extraction_index = saved["next_extraction_index"]
        last_extraction_step = saved["last_extraction_step"]
    while used_steps < spec["joint_steps"]:
        batch, batch_summaries = [], []
        while used_steps < spec["joint_steps"] and len(batch) < 16:
            seed = scenario_seed(spec["scenario_namespace"], spec["pair_seed"], used_steps)
            block_remaining = SCENARIO_BLOCK_STEPS - used_steps % SCENARIO_BLOCK_STEPS
            extraction_remaining = (extraction_steps[next_extraction_index] - used_steps
                                    if next_extraction_index < len(extraction_steps) else
                                    spec["joint_steps"] - used_steps)
            episode, summary = collect_episode(
                model, seed=seed, generator=generator,
                max_steps=min(block_remaining, spec["joint_steps"] - used_steps,
                              extraction_remaining),
                episode_index=episode_number)
            if not summary["joint_steps"]:
                raise RuntimeError("Warehouse v6 self-play rollout yielded zero steps")
            summary["start_joint_step"] = used_steps
            summary["assigned_block_seed"] = seed
            batch.append(episode)
            batch_summaries.append(summary)
            histories.append(summary)
            used_steps += summary["joint_steps"]
            episode_number += 1
            if programs is not None:
                for index, role in enumerate(AGENTS):
                    buffers[role].extend(_program_rows((episode,), index))
            if (next_extraction_index < len(extraction_steps) and
                    used_steps == extraction_steps[next_extraction_index]):
                break
        update = ppo_update(model, optimizer, batch, programs,
                            progress=used_steps / spec["joint_steps"],
                            cost_multiplier=cost_multiplier,
                            program_validation_rows=program_validation_rows)
        # Only actual, complete 0/1 episode outcomes move the risk dual.
        complete_outcomes = [row["shutdown_episode"] for row in batch_summaries
                             if row["complete"]]
        shutdown_rate = (float(np.mean(complete_outcomes))
                         if complete_outcomes else None)
        dual_report = risk_dual.update([
            (bool(row["complete"]), bool(row["shutdown_episode"]))
            for row in batch_summaries])
        cost_multiplier = risk_dual.multiplier
        update.update(joint_steps=used_steps, complete_episode_shutdown_rate=shutdown_rate,
                      complete_episode_window_shutdown_rate=dual_report[
                          "window_shutdown_rate"],
                      complete_episode_window_size=dual_report["window_episodes"],
                      completed_episodes_for_dual=len(complete_outcomes),
                      next_cost_multiplier=cost_multiplier,
                      risk_upper_bound=dual_report["one_sided_upper"],
                      risk_dual_phase=dual_report["phase"])
        updates.append(update)
        if (next_extraction_index < len(extraction_steps) and
                used_steps == extraction_steps[next_extraction_index]):
            _assert_source_closure(spec["source_closure_sha256"])
            if programs is not None:
                if spec.get("smoke"):
                    programs, extracted, program_validation_rows = \
                        _fit_training_programs(model, buffers)
                else:
                    programs, extracted, program_validation_rows = \
                        _fit_training_programs(
                            model, buffers, method=spec["program_fit_method"])
                for role in AGENTS:
                    extractions[role].append({
                        "joint_steps": used_steps,
                        "window_start_joint_step": last_extraction_step,
                        "window_end_joint_step": used_steps,
                        **extracted[role],
                    })
                    buffers[role].clear()
            last_extraction_step = used_steps
            next_extraction_index += 1
        if used_steps >= next_eval or used_steps == spec["joint_steps"]:
            _assert_source_closure(spec["source_closure_sha256"])
            measured = evaluate(model, spec["checkpoint_seeds"])
            checkpoint_result = {"joint_steps": used_steps, **measured["summary"]}
            checkpoint_result["external_feedback_rows"] = {
                role: sum(item["feedback_rows"][role] for item in updates)
                for role in AGENTS}
            checkpoint_result["external_feedback_gradient_norm"] = {
                role: sum(item["feedback_gradient_norm"][role] for item in updates)
                for role in AGENTS}
            development.append(checkpoint_result)
            selectable = variant == "raw" or used_steps / spec["joint_steps"] >= 0.85
            if variant == "regularized" and not spec.get("smoke", False):
                selectable = selectable and all(
                    checkpoint_result["external_feedback_rows"][role] > 0 and
                    checkpoint_result["external_feedback_gradient_norm"][role] > 0
                    for role in AGENTS)
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
            checkpoint_payload = {
                "training_config_sha256": spec_hash, "variant": variant,
                "source_closure_sha256": spec["source_closure_sha256"],
                "run_spec_sha256": run_spec_sha256,
                "pair_seed": spec["pair_seed"],
                "joint_steps": spec["joint_steps"],
                "initial_role_sha256": initial_role_hashes,
                "model_version": MODEL_VERSION,
                "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
                "program_fit_method": spec.get("program_fit_method"),
                "bc_recipe_sha256": spec.get("bc_recipe_sha256"),
                "bc_data_seed_schedule_sha256": spec.get(
                    "bc_data_seed_schedule_sha256"),
                "risk_dual_config_sha256": spec.get("risk_dual_config_sha256"),
                "complete": False, "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "generator_state": generator.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "used_steps": used_steps, "episode_number": episode_number,
                "histories": histories, "updates": updates,
                "development": development, "best": best,
                "cost_multiplier": cost_multiplier, "next_eval": next_eval,
                "next_extraction_index": next_extraction_index,
                "last_extraction_step": last_extraction_step,
                "risk_dual_snapshot": risk_dual.snapshot(),
                "buffers": ({role: list(buffers[role]) for role in AGENTS}
                            if programs is not None else {}),
                "programs": ({role: (programs[role].to_dict()
                                      if programs[role] is not None else None)
                              for role in AGENTS} if programs is not None else {}),
                "program_validation_rows": program_validation_rows,
                "extractions": extractions,
            }
            checkpoint_payload["state_sha256"] = _state_dict_sha256(
                checkpoint_payload["state_dict"])
            checkpoint_payload["best_state_sha256"] = (
                _state_dict_sha256(best["state_dict"]) if best is not None else None)
            checkpoint_payload["resume_progress_sha256"] = \
                _resume_progress_sha256(checkpoint_payload)
            _save_checkpoint(checkpoint, checkpoint_payload)
    if best is None:
        raise RuntimeError("Warehouse v6 training produced no selected checkpoint")
    if variant == "regularized" and spec.get("smoke", False):
        scenes = [seed for item in extractions[AGENTS[0]]
                  for values in item.get("training_split_scenario_seeds", {}).values()
                  for seed in values]
        cohorts_valid = (len(extractions[AGENTS[0]]) == len(extraction_steps)
                         and len(scenes) == len(set(scenes)))
    else:
        cohorts_valid = (online_program_cohorts_valid(
            extractions, joint_steps=spec["joint_steps"])
            if variant == "regularized" else True)
    if not cohorts_valid:
        raise RuntimeError("Warehouse online program cohorts overlap or miss a prescribed step")
    _assert_source_closure(spec["source_closure_sha256"])
    model.load_state_dict(best["state_dict"])
    actor_artifact = export_actor(
        model, directory / f"{variant}.actor.npz",
        pair_seed=spec["pair_seed"], variant=variant,
        candidate_id=spec["candidate_id"])
    online_selected_programs = {role: program for role, program in best["programs"].items()
                                if program is not None}
    if variant == "regularized" and not spec.get("smoke", False):
        selected_programs, final_program_reports = _fresh_final_programs(
            model, spec, histories, directory=directory,
            actor_artifact=actor_artifact)
    else:
        selected_programs, final_program_reports = {}, {}
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
    actual_resets = [{key: row[key] for key in (
        "seed", "episode_index", "episode_key", "start_joint_step", "joint_steps",
        "complete", "shutdown_episode")}
        for row in histories]
    report = {
        "runner_version": RUNNER_VERSION, "candidate_id": spec["candidate_id"],
        "source_closure_sha256": spec["source_closure_sha256"],
        "pair_seed": spec["pair_seed"], "variant": variant,
        "joint_steps": used_steps, "actor_decisions": 2 * used_steps,
        "model_version": MODEL_VERSION, "actor_model_version": MODEL_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "run_spec_sha256": run_spec_sha256,
        "training_config_sha256": spec_hash,
        "release_id": spec.get("release_id"),
        "architecture_mode": spec.get("architecture_mode", "smoke"),
        "program_fit_method": spec.get("program_fit_method", "smoke"),
        "bc_recipe_version": spec.get("bc_recipe_version", "smoke"),
        "bc_recipe_sha256": spec.get("bc_recipe_sha256"),
        "bc_data_seed_schedule_sha256": spec.get(
            "bc_data_seed_schedule_sha256"),
        "bc_critic_reset_schedule": spec.get("bc_critic_reset_schedule"),
        "bc_critic_reset_schedule_sha256": spec.get(
            "bc_critic_reset_schedule_sha256"),
        "risk_dual_config_sha256": spec.get("risk_dual_config_sha256"),
        "online_program_extraction_steps": list(extraction_steps),
        "online_program_split_block_cycle": dict(FINAL_PROGRAM_SPLIT_COUNTS),
        "online_program_cohorts_disjoint": cohorts_valid,
        "final_program_fit_seed_schedule": spec.get(
            "final_program_fit_seed_schedule"),
        "final_program_fit_seed_schedule_sha256": spec.get(
            "final_program_fit_seed_schedule_sha256"),
        "checkpoint_seeds": list(spec["checkpoint_seeds"]),
        "checkpoint_seed_sha256": _digest_json(spec["checkpoint_seeds"]),
        "scenario_schedule_sha256": scenario_schedule_sha256(
            spec["scenario_namespace"], spec["pair_seed"], spec["joint_steps"]),
        "assigned_scenario_schedule_kind": "fixed_120_joint_step_blocks",
        "actual_episode_resets": actual_resets,
        "actual_episode_resets_sha256": _digest_json(actual_resets),
        "initial_role_sha256": initial_role_hashes,
        "selected_role_sha256": role_hashes(model),
        "actor_artifact": actor_artifact,
        "role_weight_change_norm": _role_weight_change(initial, model),
        "post_policy_interventions": 0,
        "physical_no_legal_action_rows": sum(
            sum(item["physical_no_legal_action_rows"].values()) for item in updates),
        "feedback_rows": feedback_rows,
        "feedback_gradient_norm": gradient_norm,
        "feedback_disabled_epochs": {role: sum(
            item["feedback_disabled_epochs"][role] for item in selected_updates)
            for role in AGENTS},
        "feedback_window_actor_rows": {role: sum(
            item["feedback_window_actor_rows"][role] for item in selected_updates)
            for role in AGENTS},
        "zero_action_coverage_rows": {role: sum(
            item["zero_action_coverage_rows"][role] for item in selected_updates)
            for role in AGENTS},
        "cost_critic_training": True, "cost_target_event": "complete_episode_shutdown_binary",
        "cost_actor_advantage": "complete_episode_undiscounted_event_minus_sigmoid_critic",
        "cost_actor_ratio": "joint_product_of_two_nn_actor_ratios",
        "cost_constraint_target": risk_dual.config.target,
        "risk_dual_snapshot": risk_dual.snapshot(),
        "final_cost_multiplier": cost_multiplier,
        "training_reward_overrides": dict(TRAINING_REWARD_OVERRIDES),
        "selected_development": best["development"],
        "quality": {key: value for key, value in best["development"].items()
                    if key != "joint_steps"},
        "development": development,
        "training_episodes": histories,
        "updates": updates,
        "program_extractions": extractions,
        "online_selected_program_sha256": {
            role: program.sha256() for role, program in online_selected_programs.items()},
        "final_fresh_program_reports": final_program_reports,
        "selected_program_reports": {role: program.report
                                     for role, program in selected_programs.items()},
    }
    if variant == "raw" and (any(feedback_rows.values()) or any(gradient_norm.values())):
        raise RuntimeError("Warehouse v6 raw baseline received a program gradient")
    completed_checkpoint = {
        "training_config_sha256": spec_hash, "variant": variant,
        "source_closure_sha256": spec["source_closure_sha256"],
        "run_spec_sha256": run_spec_sha256,
        "pair_seed": spec["pair_seed"],
        "joint_steps": spec["joint_steps"],
        "initial_role_sha256": initial_role_hashes,
        "model_version": MODEL_VERSION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "program_fit_method": spec.get("program_fit_method"),
        "bc_recipe_sha256": spec.get("bc_recipe_sha256"),
        "bc_data_seed_schedule_sha256": spec.get("bc_data_seed_schedule_sha256"),
        "risk_dual_config_sha256": spec.get("risk_dual_config_sha256"),
        "complete": True, "selected_state_dict": model.state_dict(),
        "selected_programs": {role: program.to_dict()
                              for role, program in selected_programs.items()},
        "report": report,
    }
    completed_checkpoint["selected_state_sha256"] = _state_dict_sha256(
        completed_checkpoint["selected_state_dict"])
    completed_checkpoint["selected_programs_sha256"] = _digest_json(
        completed_checkpoint["selected_programs"])
    completed_checkpoint["report_sha256"] = _digest_json(report)
    _save_checkpoint(checkpoint, completed_checkpoint)
    return model, selected_programs, report


def train_pair(run_spec: Mapping) -> dict:
    """Train both variants from one BC start; write isolated, resumable runs."""
    spec = _checked_spec(run_spec)
    directory = (Path(spec["output_dir"]) / spec["candidate_id"] /
                 f"pair_{spec['pair_seed']}_steps_{spec['joint_steps']}")
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(spec["pair_seed"])
    base = WarehouseV6Actor()
    fresh_initial_role_hashes = role_hashes(base)
    bc_checkpoint = directory / "paired_bc.checkpoint.pt"
    spec_hash = _training_config_sha256(spec)
    run_spec_sha256 = _scientific_run_spec_sha256(spec)
    if bc_checkpoint.is_file():
        saved = torch.load(bc_checkpoint, map_location="cpu", weights_only=False)
        if (saved.get("training_config_sha256") != spec_hash
                or saved.get("source_closure_sha256") != spec["source_closure_sha256"]
                or (not spec.get("smoke") and (
                    saved.get("run_spec_sha256") != run_spec_sha256
                    or saved.get("pair_seed") != spec["pair_seed"]
                    or saved.get("joint_steps") != spec["joint_steps"]
                    or saved.get("risk_dual_config_sha256") !=
                    spec["risk_dual_config_sha256"]
                    or saved.get("model_version") != MODEL_VERSION
                    or saved.get("bc_recipe_sha256") != spec["bc_recipe_sha256"]
                    or saved.get("bc_data_seed_schedule_sha256") !=
                    spec["bc_data_seed_schedule_sha256"]
                    or saved.get("program_fit_method") !=
                    spec["program_fit_method"]))):
            raise RuntimeError("Warehouse v6 BC checkpoint run spec mismatch")
        if (not spec.get("smoke") and (
                saved.get("state_sha256") !=
                _state_dict_sha256(saved.get("state_dict", {}))
                or saved.get("report_sha256") !=
                _digest_json(saved.get("report")))):
            raise RuntimeError("Warehouse v6 BC checkpoint evidence changed")
        base.load_state_dict(saved["state_dict"])
        bc_report = saved["report"]
        if (bc_report["pretrained_role_sha256"] != role_hashes(base)
                or (not spec.get("smoke") and
                    bc_report.get("initial_role_sha256") !=
                    fresh_initial_role_hashes)):
            raise RuntimeError("Warehouse v6 BC checkpoint Actor hashes mismatch")
        if (not spec.get("smoke") and not bc_report_valid(
                bc_report, spec["pair_seed"], spec["source_closure_sha256"])):
            raise RuntimeError("Warehouse v6 BC checkpoint recipe mismatch")
    else:
        bc_report = pretrain_pair(base, pair_seed=spec["pair_seed"], spec=spec)
        if (not spec.get("smoke") and not bc_report_valid(
                bc_report, spec["pair_seed"], spec["source_closure_sha256"])):
            raise RuntimeError("Warehouse v6 produced an invalid BC stage receipt")
        _assert_source_closure(spec["source_closure_sha256"])
        bc_payload = {
            "training_config_sha256": spec_hash,
            "run_spec_sha256": run_spec_sha256,
            "pair_seed": spec["pair_seed"], "joint_steps": spec["joint_steps"],
            "source_closure_sha256": spec["source_closure_sha256"],
            "model_version": MODEL_VERSION,
            "program_fit_method": spec.get("program_fit_method"),
            "bc_recipe_sha256": spec.get("bc_recipe_sha256"),
            "bc_data_seed_schedule_sha256": spec.get(
                "bc_data_seed_schedule_sha256"),
            "risk_dual_config_sha256": spec.get("risk_dual_config_sha256"),
            "state_dict": base.state_dict(), "report": bc_report,
        }
        bc_payload["state_sha256"] = _state_dict_sha256(bc_payload["state_dict"])
        bc_payload["report_sha256"] = _digest_json(bc_report)
        _save_checkpoint(bc_checkpoint, bc_payload)
    initial = deepcopy(base.state_dict())
    bc_checkpoint_sha256 = sha256(bc_checkpoint.read_bytes()).hexdigest()
    variants = {}
    for variant in ("raw", "regularized"):
        _assert_source_closure(spec["source_closure_sha256"])
        model, programs, report = _train_variant(
            initial, variant=variant, spec=spec, directory=directory)
        actor_artifact = report["actor_artifact"]
        actor_path = Path(actor_artifact["path"])
        if (not actor_path.is_file()
                or sha256(actor_path.read_bytes()).hexdigest() != actor_artifact["sha256"]
                or actor_artifact["role_sha256"] != role_hashes(model)):
            raise RuntimeError("Warehouse v6 selected Actor artifact changed")
        program_artifacts = {}
        for role, program in programs.items():
            path = directory / f"{variant}.{role}.program.json"
            digest = program.save(path)
            program_artifacts[role] = {"path": str(path), "sha256": digest,
                                       "actor_role_sha256": program.actor_role_sha256}
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
        "joint_steps": spec["joint_steps"],
        "run_spec_sha256": run_spec_sha256,
        "training_config_sha256": spec_hash,
        "release_id": spec.get("release_id"),
        "architecture_mode": spec.get("architecture_mode", "smoke"),
        "program_fit_method": spec.get("program_fit_method", "smoke"),
        "online_program_extraction_steps": list(spec.get(
            "online_program_extraction_steps", ())),
        "online_program_split_block_cycle": dict(FINAL_PROGRAM_SPLIT_COUNTS),
        "online_program_cohorts_disjoint": variants["regularized"][
            "online_program_cohorts_disjoint"],
        "final_program_fit_seed_schedule": spec.get(
            "final_program_fit_seed_schedule"),
        "final_program_fit_seed_schedule_sha256": spec.get(
            "final_program_fit_seed_schedule_sha256"),
        "source_closure_sha256": spec["source_closure_sha256"],
        "source_files_sha256": source_closure_files_sha256(),
        "bc_recipe_version": bc_report.get("bc_recipe_version", "smoke"),
        "bc_recipe_sha256": spec.get("bc_recipe_sha256"),
        "bc_data_seed_schedule": spec.get("bc_data_seed_schedule"),
        "bc_data_seed_schedule_sha256": spec.get(
            "bc_data_seed_schedule_sha256"),
        "bc_critic_reset_schedule": spec.get("bc_critic_reset_schedule"),
        "bc_critic_reset_schedule_sha256": spec.get(
            "bc_critic_reset_schedule_sha256"),
        "risk_dual_config_sha256": spec.get("risk_dual_config_sha256"),
        "paired_bc_checkpoint_path": str(bc_checkpoint.resolve()),
        "paired_bc_checkpoint_sha256": bc_checkpoint_sha256,
        "bc_provenance_evidence_scope":
        "runner_receipt_and_checkpoint_consistency_not_independent_execution_proof",
        "checkpoint_seeds": list(spec["checkpoint_seeds"]),
        "assigned_scenario_schedule_sha256": variants["raw"]["scenario_schedule_sha256"],
        "actual_episode_resets_equal": bool(
            variants["raw"]["actual_episode_resets_sha256"] ==
            variants["regularized"]["actual_episode_resets_sha256"]),
        "actual_episode_reset_count": {variant: len(variants[variant]["actual_episode_resets"])
                                       for variant in ("raw", "regularized")},
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
    parser.add_argument("--release-id")
    parser.add_argument("--program-fit-method",
                        help="required for registered runs; must match the immutable protocol")
    parser.add_argument("--bc-episodes", type=int)
    parser.add_argument("--bc-epochs", type=int)
    parser.add_argument("--dagger-rounds", type=int)
    parser.add_argument("--dagger-episodes", type=int)
    parser.add_argument("--dagger-epochs", type=int)
    parser.add_argument("--smoke", action="store_true",
                        help="allow a tiny non-protocol test run")
    args = parser.parse_args()
    run_spec = {key: value for key, value in vars(args).items() if value is not None}
    if not args.smoke:
        from experiments.warehouse.protocol import PairRunSpec
        if args.release_id is None or args.program_fit_method is None:
            parser.error("registered runs require --release-id and --program-fit-method")
        run_spec = PairRunSpec(
            args.candidate_id, args.pair_seed, args.joint_steps, args.output_dir,
            args.program_fit_method,
            release_id=args.release_id,
            scenario_namespace=args.scenario_namespace,
            checkpoint_seeds=tuple(args.checkpoint_seeds)).to_mapping()
    report = train_pair(run_spec)
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
