from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch

from backend.training.warehouse import _collect_behavior_cloning_dataset
from backend.adapters.warehouse import (
    WAREHOUSE_PROGRAM_VERSION,
    WarehouseAdapter,
)
from env.warehouse.environment import WarehouseConfig, WarehouseMultiAgentEnv
from env.warehouse.mappo import (
    MAPPOConfig,
    MAPPOPolicy,
    MAPPOTrainer,
    MAPPO_TRAINING_CHECKPOINT_VERSION,
)
from env.warehouse.rewards import REWARD_VERSION
from env.warehouse.observations import global_state_dim, observation_dim
from env.warehouse.seed_calibration import (
    calibrate_parallel_seed_pairs,
    load_parallel_seed_library,
    save_parallel_seed_library,
)
from env.warehouse.navigation import ACTIONS


def _small_policy(*, horizon: int = 8) -> MAPPOPolicy:
    return MAPPOPolicy(
        WarehouseConfig(horizon=horizon),
        MAPPOConfig(hidden_dim=16, update_epochs=1, minibatch_size=64, seed=17),
    )


def test_behavior_cloning_targets_are_statically_legal_actor_actions() -> None:
    rows, targets, intents, _, _, _ = _collect_behavior_cloning_dataset(
        WarehouseConfig(horizon=24),
        sample_count=128,
        seed=702_026,
    )
    local_dim = observation_dim(WarehouseConfig(horizon=24))
    masks = rows[:, local_dim - len(ACTIONS) : local_dim]
    assert np.all(masks[np.arange(len(targets)), targets] > 0.5)
    assert intents.shape == targets.shape
    assert np.all((0 <= intents) & (intents < 5))


def test_actor_mask_does_not_preempt_teammate_occupancy_dynamics() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=8))
    environment.reset(seed=71)
    state = environment.get_state()
    state.by_id("robot_1").position = (8, 5)
    state.by_id("robot_2").position = (9, 5)
    environment.set_state(state)

    mask = environment.action_masks()["robot_1"]
    assert mask[ACTIONS.index("DOWN")] == pytest.approx(1.0)

    _, _, terminated, truncated, info = environment.step(
        {"robot_1": "DOWN", "robot_2": "WAIT"}
    )

    assert not terminated and not truncated
    assert info["robot_collision_event"]
    assert info["robot_collision_kind"] == "occupied_stationary"
    assert environment.get_state().by_id("robot_1").position == (8, 5)


def test_robot_two_neural_action_conditions_on_robot_one_current_action() -> None:
    config = WarehouseConfig(horizon=8)
    environment = WarehouseMultiAgentEnv(config)
    observations, _ = environment.reset(seed=72)
    policy = MAPPOPolicy(config, MAPPOConfig(hidden_dim=16, seed=72))
    with torch.no_grad():
        for parameter in policy.network.actor.parameters():
            parameter.zero_()
        first = policy.network.actor[0]
        second = policy.network.actor[2]
        third = policy.network.actor[4]
        output = policy.network.actor[6]
        context_offset = observation_dim(config)
        first.weight[0, context_offset + 1 + ACTIONS.index("UP")] = 1.0
        first.weight[1, context_offset + 1 + ACTIONS.index("DOWN")] = 1.0
        second.weight[0, 0] = 1.0
        second.weight[1, 1] = 1.0
        third.weight[0, 0] = 1.0
        third.weight[1, 1] = 1.0
        output.weight[ACTIONS.index("LEFT"), 0] = 8.0
        output.weight[ACTIONS.index("WAIT"), 1] = 8.0

    up_actions, _ = policy.act(
        observations,
        environment.global_state(),
        deterministic=True,
        fixed_actions={"robot_1": "UP"},
    )
    down_actions, _ = policy.act(
        observations,
        environment.global_state(),
        deterministic=True,
        fixed_actions={"robot_1": "DOWN"},
    )

    assert up_actions == {"robot_1": "UP", "robot_2": "LEFT"}
    assert down_actions == {"robot_1": "DOWN", "robot_2": "WAIT"}


def test_actor_rejects_attempt_to_override_ai_robot_action() -> None:
    config = WarehouseConfig(horizon=8)
    environment = WarehouseMultiAgentEnv(config)
    observations, _ = environment.reset(seed=90)
    policy = MAPPOPolicy(config, MAPPOConfig(hidden_dim=16, seed=90))

    with pytest.raises(ValueError, match="AI robot actions are immutable"):
        policy.act(
            observations,
            environment.global_state(),
            deterministic=True,
            fixed_actions={"robot_2": "WAIT"},
        )


def test_shared_observation_contract_and_task_slots() -> None:
    config = WarehouseConfig(horizon=8)
    environment = WarehouseMultiAgentEnv(config)
    observations, _ = environment.reset(seed=12)
    assert observation_dim(config) == 466
    assert global_state_dim(config) == 940
    assert set(observations) == {"robot_1", "robot_2"}
    assert all(value.shape == (466,) for value in observations.values())
    assert environment.global_state().shape == (940,)

    schema = WarehouseAdapter(environment).observation_schema()
    serialized = str(schema)
    assert "task" in serialized.lower()
    assert schema["contract_version"] == (
            "collaborative_observation_v23_avoidable_wait_memory"
    )
    assert "navigation_goal_fields" in schema


@pytest.mark.parametrize(
    "old_version",
    [
        "warehouse_mappo_v8_shared_two_robot_delivery_stream",
        "warehouse_mappo_v10_safe_mission_collision_recovery",
    ],
)
def test_old_checkpoint_version_is_rejected(tmp_path, old_version: str) -> None:
    policy = _small_policy()
    good = tmp_path / "good.pt"
    old = tmp_path / "old.pt"
    policy.save(good)
    payload = torch.load(good, map_location="cpu", weights_only=False)
    payload["model_version"] = old_version
    torch.save(payload, old)
    with pytest.raises(ValueError, match="Unsupported MAPPO checkpoint version"):
        MAPPOPolicy.load(old)


def test_old_reward_and_training_checkpoint_versions_are_rejected(tmp_path) -> None:
    policy = _small_policy()
    model_path = tmp_path / "wrong_reward.pt"
    policy.save(model_path)
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    payload["reward_version"] = "warehouse_path_reward_v1"
    torch.save(payload, model_path)
    with pytest.raises(ValueError, match="incompatible reward version"):
        MAPPOPolicy.load(model_path)

    checkpoint = tmp_path / "old_training.pt"
    trainer = MAPPOTrainer(policy)
    trainer.save_checkpoint(checkpoint, episode=0, metrics=[])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["checkpoint_version"] == MAPPO_TRAINING_CHECKPOINT_VERSION
    assert payload["reward_version"] == REWARD_VERSION
    payload["checkpoint_version"] = "warehouse_mappo_training_v3_collision_recovery"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="Unsupported MAPPO training checkpoint"):
        MAPPOTrainer.load_checkpoint(checkpoint)


def test_proxy_human_samples_are_actor_masked_but_critic_trained() -> None:
    policy = _small_policy(horizon=12)
    trainer = MAPPOTrainer(policy)
    environment = WarehouseMultiAgentEnv(policy.environment_config)
    batch = None
    for seed in range(200):
        candidate = trainer.collect_episode(environment, seed=seed)
        if candidate.proxy_human_overrides:
            batch = candidate
            break
    assert batch is not None, "deterministic seed search should find a proxy-human episode"
    assert batch.proxy_human_overrides > 0
    assert np.count_nonzero(batch.trainable_mask <= 0.5) >= batch.proxy_human_overrides
    metrics = trainer.update(batch)
    assert metrics["critic_masked_sample_updates"] > 0
    assert np.isfinite(metrics["critic_masked_sample_loss"])


def test_critical_skill_anchor_updates_actor_without_runtime_intervention() -> None:
    policy = _small_policy(horizon=8)
    trainer = MAPPOTrainer(policy)
    environment = WarehouseMultiAgentEnv(policy.environment_config)
    batch = trainer.collect_episode(
        environment,
        seed=812,
        coordination_curriculum_probability=1.0,
    )
    anchor_observations = batch.observations[:8].copy()
    anchor_labels = batch.actions[:8].copy()
    intent_before = [
        parameter.detach().clone()
        for parameter in policy.network.intent_encoder.parameters()
    ]

    metrics = trainer.update(
        batch,
        skill_anchor_observations=anchor_observations,
        skill_anchor_labels=anchor_labels,
        skill_anchor_weight=0.1,
    )

    assert np.isfinite(metrics["skill_anchor_loss"])
    assert metrics["skill_anchor_loss"] > 0.0
    assert 0.0 <= metrics["skill_anchor_accuracy"] <= 1.0
    assert metrics["skill_anchor_weight"] == pytest.approx(0.1)
    # The curriculum changes only the initial state. Every transition stored
    # in the batch remains the Actor action actually submitted to env.step.
    assert batch.coordination_curriculum_kind in {
        "head_on",
        "charger_handoff",
        "delivery_goal_clearance",
        "charger_commitment",
        "task_commitment",
    }
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(
            intent_before,
            policy.network.intent_encoder.parameters(),
        )
    )
    assert np.count_nonzero(batch.trainable_mask <= 0.5) == (
        batch.proxy_human_overrides
    )


def test_critical_skill_anchor_rejects_misaligned_training_rows() -> None:
    policy = _small_policy(horizon=4)
    trainer = MAPPOTrainer(policy)
    environment = WarehouseMultiAgentEnv(policy.environment_config)
    batch = trainer.collect_episode(environment, seed=813)

    with pytest.raises(ValueError, match="observations and labels must align"):
        trainer.update(
            batch,
            skill_anchor_observations=batch.observations[:2],
            skill_anchor_labels=batch.actions[:1],
            skill_anchor_weight=0.1,
        )


def test_energy_curriculum_is_training_only_and_reproducibly_logged() -> None:
    policy = _small_policy(horizon=4)
    trainer = MAPPOTrainer(policy)
    environment = WarehouseMultiAgentEnv(policy.environment_config)
    batch = trainer.collect_episode(
        environment,
        seed=808,
        energy_curriculum_probability=1.0,
        energy_curriculum_min_battery=25.0,
        energy_curriculum_max_battery=25.0,
    )
    assert batch.energy_curriculum_applied is True
    assert batch.initial_minimum_battery == pytest.approx(25.0)
    assert sorted(agent.battery for agent in environment.get_state().agents)[-1] <= 100.0

    formal_environment = WarehouseMultiAgentEnv(policy.environment_config)
    formal_environment.reset(seed=808)
    assert all(
        agent.battery == 100.0
        for agent in formal_environment.get_state().agents
    )


def test_explanation_evidence_binds_robot_two_live_task_and_frame() -> None:
    policy = _small_policy()
    environment = WarehouseMultiAgentEnv(policy.environment_config)
    environment.reset(seed=77)
    state = environment.get_state()
    robot = state.by_id("robot_2")
    task = state.tasks[0]
    robot.position = task.pickup_position
    environment.set_state(state)
    environment.step({"robot_1": "WAIT", "robot_2": "WAIT"})

    adapter = WarehouseAdapter(environment)
    snapshot = adapter.snapshot(policy)
    facts = adapter.evidence_facts(snapshot, "robot_2", policy)
    objective = next(
        fact for fact in facts if fact.fact_id == "robot_2.objective_reason"
    )
    assert objective.predicate == "shared_objective_selection_reason"
    assert objective.value["schema"] == "shared_objective_selection_reason.v2"
    assert objective.value["evidence_frame"] == environment.get_state().frame
    assert objective.value["selected_objective"]["task_id"] == task.task_id
    assert objective.value["task_state"]["carrying_task_id"] == task.task_id
    assert objective.value["active_shared_tasks"][0]["pickup_position"]


def test_program_version_constant_is_new_shared_contract() -> None:
    assert WAREHOUSE_PROGRAM_VERSION == "warehouse_rcpd_v30_individual_credit_posthoc"


def test_removed_runtime_controller_predicate_has_no_special_verbalizer() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig())
    adapter = WarehouseAdapter(environment)
    value = {
        "reason": "head_on_priority",
        "proposed_action": "RIGHT",
        "coordinated_action": "LEFT",
        "priority_agent_id": "robot_2",
        "yielding_agent_id": "robot_1",
        "priority_basis": "loaded_delivery",
        "priority_task_id": "task_2",
        "priority_action": "DOWN",
        "yielding_action": "LEFT",
    }

    rendered = adapter.explanation_verbalize_unit(
        {
            "predicate": "coordination_shield_reason",
            "arguments": ("robot_1", "RIGHT", "LEFT"),
            "value": value,
        },
        "zh-CN",
    )

    assert rendered.startswith("coordination_shield_reason:")
    assert "协作控制" not in rendered


def test_rcpd_semantic_contract_requires_shared_task_features() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=8))
    environment.reset(seed=5)
    groups = WarehouseAdapter(environment).required_program_predicate_groups()
    assert "shared_task_state" in groups
    assert "cargo_state" not in groups
    assert "self.carrying_shared_task" in groups["shared_task_state"]
    assert any(
        feature.startswith("task.slot_")
        for feature in groups["shared_task_state"]
    )


def test_parallel_seed_library_is_ai_ai_calibrated_and_versioned(tmp_path) -> None:
    policy = _small_policy(horizon=8)
    pairs = calibrate_parallel_seed_pairs(policy, range(300, 364))
    assert len(pairs) == 4
    assert all(pair.delivery_gap <= 1 for pair in pairs)
    assert all(pair.score_gap <= 50 for pair in pairs)
    path = save_parallel_seed_library(tmp_path / "pairs.json", pairs)
    loaded = load_parallel_seed_library(path)
    assert [(item.task1_seed, item.task2_seed) for item in loaded] == [
        (item.task1_seed, item.task2_seed) for item in pairs
    ]
