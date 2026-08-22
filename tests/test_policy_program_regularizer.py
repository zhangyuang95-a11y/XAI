from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.policy_program_regularizer import (
    PolicyProgramRegularizer,
    RegularizationStateBatch,
    program_complexity,
)
from core.rcpd import ExecutableProgram, ProgramNode
from env.warehouse.environment import (
    WarehouseConfig,
    WarehouseMultiAgentEnv,
    shortest_path_distance,
)
from env.warehouse.mappo import MAPPOConfig, MAPPOPolicy, MAPPOTrainer
from env.warehouse.observations import _coordination_features, observation_dim
from env.warehouse.navigation import ACTIONS
from env.warehouse.scenarios import (
    apply_charger_handoff_scenario,
    apply_critical_charger_approach_scenario,
    apply_delivery_goal_clearance_scenario,
    apply_head_on_scenario,
    apply_same_target_conflict_scenario,
)


def test_goal_clearance_curriculum_labels_joint_follow_through() -> None:
    config = WarehouseConfig(horizon=8)
    environment = WarehouseMultiAgentEnv(config)
    environment.reset(seed=819)
    apply_delivery_goal_clearance_scenario(environment, variant=17)
    state = environment.get_state()

    labels = train_module._safe_navigation_teacher_actions(environment)
    targets = environment._resolve_motion(state, labels)[0]

    assert targets["robot_1"] == state.by_id("robot_1").navigation_goal_position
    assert targets["robot_2"] != state.by_id("robot_2").position
    assert "junction_conflict" in train_module.STRONG_ACTOR_CORRECTION_CATEGORIES
from env.warehouse.policy import (
    AUTOREGRESSIVE_CONTEXT_DIM,
    autoregressive_actor_input,
)
from backend.training import warehouse as train_module
from backend.training import learner_dataset as learner_dataset_module
from backend.training.learner_replay import (
    MAXIMUM_OVERSAMPLE_FACTOR,
    REPLAY_CATEGORIES,
    fit_actor_supervised,
)
from backend.training.learner_dataset import best_unilateral_mission_action


def test_rcpd_has_no_synthetic_probe_rows_and_training_states_are_passable() -> None:
    config = WarehouseConfig(horizon=16)
    policy = MAPPOPolicy(
        config,
        MAPPOConfig(hidden_dim=16, update_epochs=1, minibatch_size=64, seed=31),
    )
    environment = WarehouseMultiAgentEnv(config)
    assert not hasattr(train_module, "_shared_task_probe_records")

    rows, labels, categories, coverage = (
        train_module._collect_learner_state_relabel_dataset(
        policy,
        config,
        sample_count=64,
        seed=811,
        )
    )
    assert rows.shape == (
        64,
        observation_dim(config) + AUTOREGRESSIVE_CONTEXT_DIM,
    )
    assert labels.shape == (64,)
    assert categories.shape == (64,)
    assert set(categories).issubset(
        set(REPLAY_CATEGORIES)
    )
    assert coverage["head_on_rows"] > 0
    assert coverage["counterfactual_teammate_rows"] > 0


def test_relabel_schedule_is_independent_of_episode_batch_size() -> None:
    assert train_module._next_interval_boundary(0, 100) == 100
    assert train_module._next_interval_boundary(96, 100) == 100
    assert train_module._next_interval_boundary(104, 100) == 200
    assert train_module._next_interval_boundary(400, 10) == 410
    assert train_module._next_interval_boundary(10, 0) is None


def test_learner_state_relabel_rollouts_submit_only_actor_actions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = WarehouseConfig(horizon=8)
    policy = MAPPOPolicy(
        config,
        MAPPOConfig(hidden_dim=16, update_epochs=1, minibatch_size=32, seed=32),
    )
    submitted: list[dict[str, str]] = []
    original_step = WarehouseMultiAgentEnv.step

    def recording_step(
        environment: WarehouseMultiAgentEnv,
        actions: dict[str, str],
    ):
        submitted.append(dict(actions))
        return original_step(environment, actions)

    monkeypatch.setattr(WarehouseMultiAgentEnv, "step", recording_step)
    policy.act = lambda *_args, **_kwargs: (  # type: ignore[method-assign]
        {"robot_1": "WAIT", "robot_2": "WAIT"},
        {},
    )

    _, _, categories, coverage = train_module._collect_learner_state_relabel_dataset(
        policy,
        config,
        sample_count=32,
        seed=812,
    )

    assert submitted
    assert all(
        actions == {"robot_1": "WAIT", "robot_2": "WAIT"}
        for actions in submitted
    )
    assert coverage["policy_mismatch_rows"] > 0
    # WAIT/WAIT disagreements stay in the dedicated joint-wait bucket and
    # receive its escape margin. Ordinary mismatches are not globally forced,
    # because that causes poor Actor-state generalization.
    assert set(categories) == {"joint_wait"}


def test_targeted_detour_miner_labels_mistake_but_executes_actor_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = WarehouseConfig(horizon=8)
    environment = WarehouseMultiAgentEnv(config)
    original_reset = environment.reset
    original_step = environment.step
    submitted: list[dict[str, str]] = []

    def scenario_reset(*, seed: int | None = None):
        _, info = original_reset(seed=seed)
        apply_head_on_scenario(environment, reverse=False)
        return environment.observations(), info

    def recording_step(actions: dict[str, str]):
        submitted.append(dict(actions))
        return original_step(actions)

    monkeypatch.setattr(environment, "reset", scenario_reset)
    monkeypatch.setattr(environment, "step", recording_step)
    monkeypatch.setattr(
        learner_dataset_module,
        "WarehouseMultiAgentEnv",
        lambda _config: environment,
    )
    policy = SimpleNamespace(
        act=lambda *_args, **_kwargs: (
            {"robot_1": "UP", "robot_2": "WAIT"},
            {},
        )
    )

    rows, labels, categories, coverage = (
        train_module._collect_loaded_detour_correction_dataset(
            policy,
            config,
            sample_count=1,
            maximum_episodes=1,
            seed=913,
        )
    )

    assert rows.shape == (
        1,
        observation_dim(config) + AUTOREGRESSIVE_CONTEXT_DIM,
    )
    assert labels.shape == (1,)
    assert ACTIONS[int(labels[0])] != "UP"
    assert categories.tolist() == ["loaded_detour"]
    assert coverage["detour_correction_rows"] == 1
    assert coverage["detour_expert_actions_submitted"] == 0
    assert submitted
    assert all(
        actions == {"robot_1": "UP", "robot_2": "WAIT"}
        for actions in submitted
    )


def test_collision_miner_labels_pair_but_executes_actor_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = WarehouseConfig(horizon=4)
    environment = WarehouseMultiAgentEnv(config)
    original_reset = environment.reset
    original_step = environment.step
    submitted: list[dict[str, str]] = []

    def scenario_reset(*, seed: int | None = None):
        _, info = original_reset(seed=seed)
        apply_same_target_conflict_scenario(environment, variant=0)
        return environment.observations(), info

    def recording_step(actions: dict[str, str]):
        submitted.append(dict(actions))
        return original_step(actions)

    monkeypatch.setattr(environment, "reset", scenario_reset)
    monkeypatch.setattr(environment, "step", recording_step)
    monkeypatch.setattr(
        learner_dataset_module,
        "WarehouseMultiAgentEnv",
        lambda _config: environment,
    )
    monkeypatch.setattr(
        learner_dataset_module,
        "apply_same_target_conflict_scenario",
        lambda target, *, variant: apply_same_target_conflict_scenario(
            target,
            variant=0,
        ),
    )
    actor_collision = {"robot_1": "RIGHT", "robot_2": "UP"}
    policy = SimpleNamespace(
        act=lambda *_args, **_kwargs: (dict(actor_collision), {})
    )

    rows, labels, categories, coverage = (
        train_module._collect_actor_collision_correction_dataset(
            policy,
            config,
            sample_count=2,
            maximum_episodes=1,
            seed=914,
        )
    )

    assert rows.shape == (
        2,
        observation_dim(config) + AUTOREGRESSIVE_CONTEXT_DIM,
    )
    assert labels.shape == (2,)
    assert categories.tolist() == ["collision", "collision"]
    assert coverage["unique_predicted_collision_states"] == 1
    assert coverage["collision_correction_rows"] == 2
    assert coverage["collision_expert_actions_submitted"] == 0
    assert submitted == [actor_collision]


def test_commitment_failure_miner_replays_causal_rows_without_intervention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = WarehouseConfig(horizon=8)
    environment = WarehouseMultiAgentEnv(config)
    original_reset = environment.reset
    original_step = environment.step
    submitted: list[dict[str, str]] = []

    def scenario_reset(*, seed: int | None = None):
        _, info = original_reset(seed=seed)
        state = environment.get_state()
        state.frame = 41
        state.tasks[0].created_frame = 0
        state.tasks[1].created_frame = 40
        robot_one = state.by_id("robot_1")
        robot_one.position = environment.layout.charger_position
        robot_one.battery = 20.0
        state.by_id("robot_2").position = (1, 0)
        environment.set_state(state)
        return environment.observations(), info

    def recording_step(actions: dict[str, str]):
        submitted.append(dict(actions))
        return original_step(actions)

    monkeypatch.setattr(environment, "reset", scenario_reset)
    monkeypatch.setattr(environment, "step", recording_step)
    monkeypatch.setattr(
        learner_dataset_module,
        "WarehouseMultiAgentEnv",
        lambda _config: environment,
    )
    actor_departure = {"robot_1": "UP", "robot_2": "WAIT"}
    policy = SimpleNamespace(
        act=lambda *_args, **_kwargs: (dict(actor_departure), {})
    )

    rows, labels, categories, coverage = (
        train_module._collect_actor_commitment_failure_dataset(
            policy,
            config,
            charger_cycle_samples=2,
            task_starvation_samples=2,
            maximum_episodes=1,
            seed=915,
        )
    )

    assert rows.shape == (
        2,
        observation_dim(config) + AUTOREGRESSIVE_CONTEXT_DIM,
    )
    assert labels.shape == (2,)
    assert categories.tolist().count("charger_cycle") == 1
    assert categories.tolist().count("task_starvation") == 1
    assert coverage["premature_departures_found"] == 1
    assert coverage["starving_tasks_found"] == 1
    assert coverage["commitment_expert_actions_submitted"] == 0
    assert submitted == [actor_departure]


def test_commitment_curriculum_covers_energy_and_old_task_geometry() -> None:
    config = WarehouseConfig(horizon=16)
    rows, labels, categories, coverage = (
        train_module._collect_commitment_curriculum_dataset(
            config,
            sample_count=128,
            seed=3915,
        )
    )

    assert rows.shape == (
        128,
        observation_dim(config) + AUTOREGRESSIVE_CONTEXT_DIM,
    )
    assert labels.shape == (128,)
    assert set(categories) == {"charger_cycle", "task_starvation"}
    assert coverage["charger_commitment_curriculum_rows"] > 0
    assert coverage["task_starvation_curriculum_rows"] > 0
    assert (
        coverage[
            "commitment_curriculum_expert_actions_submitted_to_runtime"
        ]
        == 0
    )


def test_wait_stall_gets_offline_progress_label_without_action_intervention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = WarehouseConfig(horizon=8)
    policy = MAPPOPolicy(
        config,
        MAPPOConfig(hidden_dim=16, update_epochs=1, minibatch_size=32, seed=132),
    )
    submitted: list[dict[str, str]] = []
    original_step = WarehouseMultiAgentEnv.step

    def recording_step(
        environment: WarehouseMultiAgentEnv,
        actions: dict[str, str],
    ):
        submitted.append(dict(actions))
        return original_step(environment, actions)

    monkeypatch.setattr(WarehouseMultiAgentEnv, "step", recording_step)
    policy.act = lambda *_args, **_kwargs: (  # type: ignore[method-assign]
        {"robot_1": "WAIT", "robot_2": "WAIT"},
        {},
    )

    _, labels, categories, _ = train_module._collect_learner_state_relabel_dataset(
        policy,
        config,
        sample_count=32,
        seed=1812,
    )

    joint_wait_labels = labels[categories == "joint_wait"]
    assert len(joint_wait_labels) > 0
    assert np.any(joint_wait_labels != ACTIONS.index("WAIT"))
    assert submitted
    assert all(
        actions == {"robot_1": "WAIT", "robot_2": "WAIT"}
        for actions in submitted
    )


def test_unilateral_mission_correction_prefers_delivery_progress() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig())
    environment.reset(seed=100091)
    state = environment.get_state()
    carried, available = state.tasks
    carried.status = "carried"
    carried.carrier_agent_id = "robot_1"
    carried.delivery_position = (7, 3)
    available.status = "carried"
    available.carrier_agent_id = "robot_2"
    available.delivery_position = (2, 8)
    robot_one = state.by_id("robot_1")
    robot_one.position = (7, 5)
    robot_one.battery = 34.0
    robot_one.carrying_task_id = carried.task_id
    robot_two = state.by_id("robot_2")
    robot_two.position = (7, 4)
    robot_two.battery = 38.0
    robot_two.carrying_task_id = available.task_id
    environment.set_state(state)
    state = environment.get_state()

    correction = best_unilateral_mission_action(
        environment,
        state,
        {"robot_1": "UP", "robot_2": "DOWN"},
        agent_id="robot_1",
    )
    targets = environment._resolve_motion(
        state,
        {"robot_1": correction, "robot_2": "DOWN"},
    )[0]

    assert correction == "LEFT"
    assert shortest_path_distance(
        targets["robot_1"],
        state.by_id("robot_1").navigation_goal_position,
    ) < shortest_path_distance(
        state.by_id("robot_1").position,
        state.by_id("robot_1").navigation_goal_position,
    )


def test_learner_state_replay_balances_rare_collision_rows() -> None:
    config = WarehouseConfig(horizon=8)
    policy = MAPPOPolicy(
        config,
        MAPPOConfig(hidden_dim=16, update_epochs=1, minibatch_size=32, seed=33),
    )
    args = SimpleNamespace(learner_state_relabel_replay_capacity=12)
    relabeler = train_module._LearnerStateRelabeler(policy, config, args)
    relabeler.replay.rows = np.arange(8, dtype=np.float32).reshape(8, 1)
    relabeler.replay.labels = np.asarray([0, 0, 0, 0, 0, 0, 1, 1])
    relabeler.replay.categories = np.asarray(
        ["ordinary"] * 6 + ["collision"] * 2,
        dtype="<U32",
    )

    rows, labels, categories = relabeler.replay.balanced(seed=913)

    assert rows.shape == (12, 1)
    assert np.sum(labels == 0) == 6
    assert np.sum(labels == 1) == 6
    assert np.sum(categories == "ordinary") == 6
    assert np.sum(categories == "collision") == 6


def test_collision_and_junction_rows_receive_strong_actor_correction() -> None:
    categories = np.asarray(
        [
            "collision",
            "junction_conflict",
            "loaded_detour",
            "charger_cycle",
            "task_starvation",
            "joint_wait",
            "ordinary",
        ],
        dtype="<U32",
    )

    assert train_module._strong_actor_correction_mask(categories).tolist() == [
        True,
        True,
        True,
        True,
        True,
        False,
        False,
    ]


def test_category_replay_preserves_old_and_new_geometry_rows() -> None:
    replay = train_module.CategoryBalancedReplay(capacity=20)
    replay.append(
        np.arange(8, dtype=np.float32).reshape(8, 1),
        np.arange(8, dtype=np.int64),
        np.asarray(["ordinary"] * 8, dtype="<U32"),
    )

    assert replay.rows is not None
    assert replay.rows[:, 0].tolist() == list(map(float, range(8)))


def test_category_replay_retains_charger_queue_and_critical_energy_rows() -> None:
    replay = train_module.CategoryBalancedReplay(capacity=20)
    replay.append(
        np.arange(24, dtype=np.float32).reshape(24, 1),
        np.arange(24, dtype=np.int64),
        np.asarray(
            ["ordinary"] * 8
            + ["charger_queue"] * 8
            + ["critical_energy"] * 8,
            dtype="<U32",
        ),
    )

    assert replay.category_counts()["charger_queue"] == 6
    assert replay.category_counts()["critical_energy"] == 6
    assert replay.categories is not None
    assert set(replay.categories) == {
        "ordinary",
        "charger_queue",
        "critical_energy",
    }


def test_rare_replay_category_has_bounded_oversampling() -> None:
    replay = train_module.CategoryBalancedReplay(capacity=1000)
    replay.append(
        np.arange(101, dtype=np.float32).reshape(101, 1),
        np.zeros(101, dtype=np.int64),
        np.asarray(
            ["ordinary"] * 100 + ["loaded_detour"],
            dtype="<U32",
        ),
    )

    _, _, categories = replay.balanced(seed=924)

    assert np.sum(categories == "loaded_detour") == MAXIMUM_OVERSAMPLE_FACTOR
    assert np.sum(categories == "ordinary") == 100


@pytest.mark.parametrize("variant", range(208))
def test_same_target_curriculum_state_has_collision_free_teacher_label(
    variant: int,
) -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=8))
    environment.reset(seed=920 + variant)
    apply_same_target_conflict_scenario(environment, variant=variant)

    state = environment.get_state()
    teacher = train_module.stable_coordination_actions(environment)
    _, _, _, collision, _, intended = environment._resolve_motion(
        state,
        teacher,
    )

    assert not collision
    assert len(set(intended.values())) == 2


@pytest.mark.parametrize("variant", range(60))
def test_critical_charger_approach_has_safe_training_label(
    variant: int,
) -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=8))
    environment.reset(seed=1200 + variant)
    approaching_agent_id = environment.agent_ids[variant % 2]
    apply_critical_charger_approach_scenario(
        environment,
        approaching_agent_id=approaching_agent_id,
        variant=variant,
    )

    state = environment.get_state()
    approaching = state.by_id(approaching_agent_id)
    teacher = train_module.stable_coordination_actions(environment)
    _, _, _, collision, _, _ = environment._resolve_motion(state, teacher)

    assert approaching.navigation_goal_kind == "charge"
    assert teacher[approaching_agent_id] != "WAIT"
    assert not collision


def test_low_battery_charger_occupant_charges_while_teammate_queues_safely() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=8))
    environment.reset(seed=940)
    apply_charger_handoff_scenario(
        environment,
        occupant_agent_id="robot_2",
        occupant_battery=2.0,
        queued_battery=18.0,
        occupant_carrying=True,
        queued_carrying=True,
    )

    before = environment.get_state()
    teacher = train_module.stable_coordination_actions(environment)
    targets, _, invalid, collision, _, _ = environment._resolve_motion(
        before,
        teacher,
    )

    assert teacher["robot_2"] == "WAIT"
    assert teacher["robot_1"] == "WAIT"
    assert not invalid
    assert not collision
    assert targets["robot_1"] == before.by_id("robot_1").position


def test_offline_teacher_clears_a_delivery_goal_without_blocking_the_loaded_robot() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig())
    environment.reset(seed=100)
    state = environment.get_state()
    delivery_task, available_task = state.tasks
    delivery_task.task_id = "task_4"
    delivery_task.status = "carried"
    delivery_task.carrier_agent_id = "robot_2"
    delivery_task.pickup_position = (8, 9)
    delivery_task.delivery_position = (2, 5)
    available_task.task_id = "task_5"
    available_task.status = "available"
    available_task.carrier_agent_id = None
    available_task.pickup_position = (6, 7)
    available_task.delivery_position = (3, 2)
    robot_1 = state.by_id("robot_1")
    robot_1.position = (2, 5)
    robot_1.battery = 40.0
    robot_1.carrying_task_id = None
    robot_2 = state.by_id("robot_2")
    robot_2.position = (4, 5)
    robot_2.battery = 32.0
    robot_2.carrying_task_id = delivery_task.task_id
    environment.set_state(state)

    teacher = train_module.stable_coordination_actions(environment)
    before = environment.get_state()
    targets = environment._resolve_motion(before, teacher)[0]

    assert teacher == {"robot_1": "RIGHT", "robot_2": "UP"}
    assert targets["robot_1"] != before.by_id("robot_2").navigation_goal_position
    assert shortest_path_distance(
        targets["robot_2"],
        before.by_id("robot_2").navigation_goal_position,
    ) < shortest_path_distance(
        before.by_id("robot_2").position,
        before.by_id("robot_2").navigation_goal_position,
    )


def test_non_wait_margin_teaches_actor_to_leave_an_ineffective_wait() -> None:
    config = WarehouseConfig(horizon=8)
    environment = WarehouseMultiAgentEnv(config)
    observations, _ = environment.reset(seed=914)
    policy = MAPPOPolicy(
        config,
        MAPPOConfig(hidden_dim=16, update_epochs=1, minibatch_size=32, seed=34),
    )
    mask = environment.action_masks()["robot_1"]
    target_action = next(
        action
        for action, allowed in zip(ACTIONS, mask)
        if action != "WAIT" and allowed > 0.5
    )
    row = autoregressive_actor_input(
        observations["robot_1"], preceding_action=None
    )
    rows = np.repeat(row[None, :], 32, axis=0)
    labels = np.full(32, ACTIONS.index(target_action), dtype=np.int64)

    result = fit_actor_supervised(
        policy,
        rows,
        labels,
        epochs=20,
        batch_size=32,
        learning_rate=0.005,
        non_wait_margin=1.0,
        non_wait_weight=1.0,
        escape_wait_margin=2.0,
        escape_wait_weight=1.0,
        escape_wait_mask=np.zeros(len(rows), dtype=bool),
        correction_margin=1.5,
        correction_weight=1.0,
        correction_mask=np.zeros(len(rows), dtype=bool),
        wait_margin=1.0,
        wait_weight=1.0,
        wait_margin_mask=np.zeros(len(rows), dtype=bool),
        seed=915,
    )
    logits = policy.masked_actor_logits(
        torch.as_tensor(rows[:1], dtype=torch.float32)
    )[0]

    assert result["non_wait_margin"] == 1.0
    assert logits[ACTIONS.index(target_action)] > logits[ACTIONS.index("WAIT")]


def test_supervised_actor_fit_updates_the_neural_intent_encoder() -> None:
    config = WarehouseConfig(horizon=8)
    environment = WarehouseMultiAgentEnv(config)
    observations, _ = environment.reset(seed=1914)
    policy = MAPPOPolicy(
        config,
        MAPPOConfig(hidden_dim=16, update_epochs=1, minibatch_size=32, seed=134),
    )
    mask = environment.action_masks()["robot_1"]
    target_action = next(
        action
        for action, allowed in zip(ACTIONS, mask)
        if action != "WAIT" and allowed > 0.5
    )
    row = autoregressive_actor_input(
        observations["robot_1"], preceding_action=None
    )
    rows = np.repeat(row[None, :], 32, axis=0)
    labels = np.full(32, ACTIONS.index(target_action), dtype=np.int64)
    intent_before = tuple(
        parameter.detach().clone()
        for parameter in policy.network.intent_encoder.parameters()
    )

    fit_actor_supervised(
        policy,
        rows,
        labels,
        epochs=5,
        batch_size=32,
        learning_rate=0.005,
        non_wait_margin=0.0,
        non_wait_weight=0.0,
        escape_wait_margin=0.0,
        escape_wait_weight=0.0,
        escape_wait_mask=np.zeros(len(rows), dtype=bool),
        correction_margin=0.0,
        correction_weight=0.0,
        correction_mask=np.zeros(len(rows), dtype=bool),
        wait_margin=0.0,
        wait_weight=0.0,
        wait_margin_mask=np.zeros(len(rows), dtype=bool),
        seed=1915,
    )

    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(
            intent_before,
            policy.network.intent_encoder.parameters(),
        )
    )


def test_structured_relabel_scope_preserves_base_actor_and_intent() -> None:
    config = WarehouseConfig(horizon=8)
    environment = WarehouseMultiAgentEnv(config)
    observations, _ = environment.reset(seed=2214)
    policy = MAPPOPolicy(
        config,
        MAPPOConfig(hidden_dim=16, update_epochs=1, minibatch_size=32, seed=224),
    )
    row = autoregressive_actor_input(
        observations["robot_1"], preceding_action=None
    )
    rows = np.repeat(row[None, :], 32, axis=0)
    labels = np.full(32, ACTIONS.index("UP"), dtype=np.int64)
    frozen_before = tuple(
        parameter.detach().clone()
        for module in (
            policy.network.intent_encoder,
            policy.network.actor,
        )
        for parameter in module.parameters()
    )
    scorer_before = tuple(
        parameter.detach().clone()
        for parameter in policy.network.action_scorer.parameters()
    )

    fit_actor_supervised(
        policy,
        rows,
        labels,
        epochs=5,
        batch_size=32,
        learning_rate=0.005,
        non_wait_margin=0.0,
        non_wait_weight=0.0,
        escape_wait_margin=0.0,
        escape_wait_weight=0.0,
        escape_wait_mask=np.zeros(len(rows), dtype=bool),
        correction_margin=0.0,
        correction_weight=0.0,
        correction_mask=np.zeros(len(rows), dtype=bool),
        wait_margin=0.0,
        wait_weight=0.0,
        wait_margin_mask=np.zeros(len(rows), dtype=bool),
        seed=2215,
        parameter_scope="structured",
    )

    frozen_after = tuple(
        parameter.detach()
        for module in (
            policy.network.intent_encoder,
            policy.network.actor,
        )
        for parameter in module.parameters()
    )
    assert all(torch.equal(before, after) for before, after in zip(frozen_before, frozen_after))
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(
            scorer_before,
            policy.network.action_scorer.parameters(),
        )
    )


def test_supervised_actor_fit_updates_structured_neural_action_modules() -> None:
    config = WarehouseConfig(horizon=8)
    environment = WarehouseMultiAgentEnv(config)
    observations, _ = environment.reset(seed=2914)
    policy = MAPPOPolicy(
        config,
        MAPPOConfig(hidden_dim=16, update_epochs=1, minibatch_size=32, seed=234),
    )
    rows = np.stack(
        [
            autoregressive_actor_input(
                observations["robot_1"], preceding_action=None
            ),
            autoregressive_actor_input(
                observations["robot_2"], preceding_action="UP"
            ),
        ]
        * 16
    )
    labels = np.asarray(
        [ACTIONS.index("UP"), ACTIONS.index("LEFT")] * 16,
        dtype=np.int64,
    )
    scorer_before = tuple(
        parameter.detach().clone()
        for parameter in policy.network.action_scorer.parameters()
    )
    predictor_before = tuple(
        parameter.detach().clone()
        for parameter in policy.network.teammate_action_predictor.parameters()
    )

    fit_actor_supervised(
        policy,
        rows,
        labels,
        epochs=10,
        batch_size=32,
        learning_rate=0.005,
        non_wait_margin=0.0,
        non_wait_weight=0.0,
        escape_wait_margin=0.0,
        escape_wait_weight=0.0,
        escape_wait_mask=np.zeros(len(rows), dtype=bool),
        correction_margin=0.0,
        correction_weight=0.0,
        correction_mask=np.zeros(len(rows), dtype=bool),
        wait_margin=0.0,
        wait_weight=0.0,
        wait_margin_mask=np.zeros(len(rows), dtype=bool),
        seed=2915,
    )

    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(
            scorer_before,
            policy.network.action_scorer.parameters(),
        )
    )
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(
            predictor_before,
            policy.network.teammate_action_predictor.parameters(),
        )
    )


def test_joint_wait_escape_margin_strongly_separates_motion_from_wait() -> None:
    config = WarehouseConfig(horizon=8)
    environment = WarehouseMultiAgentEnv(config)
    observations, _ = environment.reset(seed=918)
    policy = MAPPOPolicy(
        config,
        MAPPOConfig(hidden_dim=16, update_epochs=1, minibatch_size=32, seed=36),
    )
    mask = environment.action_masks()["robot_1"]
    target_action = next(
        action
        for action, allowed in zip(ACTIONS, mask)
        if action != "WAIT" and allowed > 0.5
    )
    row = autoregressive_actor_input(
        observations["robot_1"], preceding_action=None
    )
    rows = np.repeat(row[None, :], 32, axis=0)
    labels = np.full(32, ACTIONS.index(target_action), dtype=np.int64)

    result = fit_actor_supervised(
        policy,
        rows,
        labels,
        epochs=30,
        batch_size=32,
        learning_rate=0.005,
        non_wait_margin=0.0,
        non_wait_weight=0.0,
        escape_wait_margin=2.0,
        escape_wait_weight=1.0,
        escape_wait_mask=np.ones(len(rows), dtype=bool),
        correction_margin=1.5,
        correction_weight=1.0,
        correction_mask=np.zeros(len(rows), dtype=bool),
        wait_margin=0.0,
        wait_weight=0.0,
        wait_margin_mask=np.zeros(len(rows), dtype=bool),
        seed=919,
    )
    logits = policy.masked_actor_logits(
        torch.as_tensor(rows[:1], dtype=torch.float32)
    )[0]
    gap = logits[ACTIONS.index(target_action)] - logits[ACTIONS.index("WAIT")]

    assert result["escape_wait_rows"] == len(rows)
    assert result["escape_wait_margin"] == 2.0
    assert gap > 1.0


def test_loaded_detour_correction_margin_beats_every_alternative_action() -> None:
    config = WarehouseConfig(horizon=8)
    environment = WarehouseMultiAgentEnv(config)
    observations, _ = environment.reset(seed=921)
    policy = MAPPOPolicy(
        config,
        MAPPOConfig(hidden_dim=16, update_epochs=1, minibatch_size=32, seed=37),
    )
    mask = environment.action_masks()["robot_1"]
    target_action = next(
        action
        for action, allowed in zip(ACTIONS, mask)
        if action != "WAIT" and allowed > 0.5
    )
    row = autoregressive_actor_input(
        observations["robot_1"], preceding_action=None
    )
    rows = np.repeat(row[None, :], 32, axis=0)
    labels = np.full(32, ACTIONS.index(target_action), dtype=np.int64)

    result = fit_actor_supervised(
        policy,
        rows,
        labels,
        epochs=30,
        batch_size=32,
        learning_rate=0.005,
        non_wait_margin=0.0,
        non_wait_weight=0.0,
        escape_wait_margin=0.0,
        escape_wait_weight=0.0,
        escape_wait_mask=np.zeros(len(rows), dtype=bool),
        correction_margin=1.5,
        correction_weight=1.0,
        correction_mask=np.ones(len(rows), dtype=bool),
        wait_margin=0.0,
        wait_weight=0.0,
        wait_margin_mask=np.zeros(len(rows), dtype=bool),
        seed=922,
    )
    logits = policy.masked_actor_logits(
        torch.as_tensor(rows[:1], dtype=torch.float32)
    )[0]
    target_index = ACTIONS.index(target_action)
    alternatives = torch.cat((logits[:target_index], logits[target_index + 1 :]))

    assert result["correction_rows"] == len(rows)
    assert result["correction_margin"] == 1.5
    assert logits[target_index] - alternatives.max() > 0.75


def test_wait_margin_teaches_actor_to_keep_charging_at_critical_battery() -> None:
    config = WarehouseConfig(horizon=8)
    environment = WarehouseMultiAgentEnv(config)
    environment.reset(seed=916)
    apply_charger_handoff_scenario(
        environment,
        occupant_agent_id="robot_1",
        occupant_battery=2.0,
        queued_battery=18.0,
        occupant_carrying=True,
        queued_carrying=True,
    )
    observations = environment.observations()
    policy = MAPPOPolicy(
        config,
        MAPPOConfig(hidden_dim=16, update_epochs=1, minibatch_size=32, seed=35),
    )
    row = autoregressive_actor_input(
        observations["robot_1"], preceding_action=None
    )
    rows = np.repeat(row[None, :], 32, axis=0)
    labels = np.full(32, ACTIONS.index("WAIT"), dtype=np.int64)

    result = fit_actor_supervised(
        policy,
        rows,
        labels,
        epochs=20,
        batch_size=32,
        learning_rate=0.005,
        non_wait_margin=1.0,
        non_wait_weight=1.0,
        escape_wait_margin=2.0,
        escape_wait_weight=1.0,
        escape_wait_mask=np.zeros(len(rows), dtype=bool),
        correction_margin=1.5,
        correction_weight=1.0,
        correction_mask=np.zeros(len(rows), dtype=bool),
        wait_margin=1.0,
        wait_weight=1.0,
        wait_margin_mask=np.ones(len(rows), dtype=bool),
        seed=917,
    )
    logits = policy.masked_actor_logits(
        torch.as_tensor(rows[:1], dtype=torch.float32)
    )[0]
    wait_logit = logits[ACTIONS.index("WAIT")]
    strongest_non_wait_logit = torch.cat(
        (logits[: ACTIONS.index("WAIT")], logits[ACTIONS.index("WAIT") + 1 :])
    ).max()

    assert result["wait_margin"] == 1.0
    assert result["wait_margin_rows"] == len(rows)
    assert wait_logit > strongest_non_wait_logit


def test_corridor_priority_matches_loaded_distance_rule_from_both_views() -> None:
    config = WarehouseConfig()
    environment = WarehouseMultiAgentEnv(config)
    environment.reset(seed=919)
    train_module._configure_learner_state_head_on(environment, reverse=False)
    state = environment.get_state()
    robot_one = _coordination_features(state, "robot_1", config)
    robot_two = _coordination_features(state, "robot_2", config)
    assert robot_one[6:8] == [1.0, 0.0]
    assert robot_two[6:8] == [0.0, 1.0]

    environment.reset(seed=920)
    train_module._configure_learner_state_head_on(environment, reverse=True)
    state = environment.get_state()
    robot_one = _coordination_features(state, "robot_1", config)
    robot_two = _coordination_features(state, "robot_2", config)
    assert robot_one[6:8] == [0.0, 1.0]
    assert robot_two[6:8] == [1.0, 0.0]


def test_periodic_head_on_evaluation_uses_current_alternating_shelf_map() -> None:
    config = WarehouseConfig(horizon=16)
    policy = MAPPOPolicy(
        config,
        MAPPOConfig(hidden_dim=16, update_epochs=1, minibatch_size=32, seed=44),
    )
    scripted_actions = iter(
        (
            {"robot_1": "DOWN", "robot_2": "UP"},
            {"robot_1": "UP", "robot_2": "DOWN"},
        )
    )
    policy.act = lambda *_args, **_kwargs: (next(scripted_actions), {})  # type: ignore[method-assign]

    result = train_module.evaluate_head_on_yield_scenarios(
        policy,
        config,
        episodes=2,
        seed=4_400,
    )

    assert result["episodes"] == 2.0
    assert result["success_rate"] == 0.0


def _program() -> ExecutableProgram:
    return ExecutableProgram(
        action_names=("A", "B"),
        feature_names=("self.x", "goal.distance"),
        root=ProgramNode(
            feature="self.x",
            threshold=0.5,
            left=ProgramNode(probabilities=(0.9, 0.1)),
            right=ProgramNode(
                feature="goal.distance",
                threshold=2.0,
                left=ProgramNode(probabilities=(0.2, 0.8)),
                right=ProgramNode(probabilities=(0.6, 0.4)),
            ),
        ),
        metadata={"regularization_version": True},
    )


def test_extracted_program_probabilities_are_valid() -> None:
    program = _program()
    for features in (
        {"self.x": 0.0, "goal.distance": 1.0},
        {"self.x": 1.0, "goal.distance": 1.0},
        {"self.x": 1.0, "goal.distance": 3.0},
    ):
        probabilities = program.predict_proba(features)
        assert set(probabilities) == {"A", "B"}
        assert all(0.0 <= value <= 1.0 for value in probabilities.values())
        assert sum(probabilities.values()) == pytest.approx(1.0)


def test_forward_kl_is_lower_when_nn_and_program_agree() -> None:
    regularizer = PolicyProgramRegularizer()
    program_probabilities = torch.tensor([[0.9, 0.1]], dtype=torch.float32)
    agreeing_logits = torch.log(program_probabilities).requires_grad_()
    disagreeing_logits = torch.log(
        torch.tensor([[0.1, 0.9]], dtype=torch.float32)
    ).requires_grad_()

    agreeing = regularizer.compute_fidelity_loss(
        agreeing_logits,
        program_probabilities,
        states=None,
    )
    disagreeing = regularizer.compute_fidelity_loss(
        disagreeing_logits,
        program_probabilities,
        states=None,
    )

    assert agreeing.item() == pytest.approx(0.0, abs=1e-6)
    assert disagreeing.item() > agreeing.item()
    disagreeing.backward()
    assert disagreeing_logits.grad is not None


def test_complexity_loss_counts_depth_leaves_and_predicates() -> None:
    complexity = program_complexity(
        _program(),
        max_depth=4,
        max_leaf_count=6,
        max_predicate_count=4,
    )

    assert complexity.depth == 2
    assert complexity.leaves == 3
    assert complexity.predicates == 2
    assert complexity.normalized_depth == pytest.approx(0.5)
    assert complexity.normalized_leaves == pytest.approx(0.5)
    assert complexity.normalized_predicates == pytest.approx(0.5)
    assert complexity.loss == pytest.approx(0.5)


def test_lambda_extract_zero_matches_baseline_actor_update() -> None:
    environment_config = WarehouseConfig(horizon=3)
    algorithm_config = MAPPOConfig(
        hidden_dim=16,
        update_epochs=1,
        minibatch_size=10_000,
        seed=1701,
    )
    baseline_policy = MAPPOPolicy(environment_config, algorithm_config)
    regularized_policy = MAPPOPolicy(environment_config, algorithm_config)
    regularized_policy.network.load_state_dict(
        baseline_policy.network.state_dict()
    )
    baseline_trainer = MAPPOTrainer(baseline_policy)
    regularized_trainer = MAPPOTrainer(regularized_policy)
    batch = baseline_trainer.collect_episode(
        WarehouseMultiAgentEnv(environment_config),
        seed=1701,
    )
    regularized_batch = deepcopy(batch)
    with torch.no_grad():
        observations = torch.as_tensor(
            batch.observations,
            dtype=torch.float32,
        )
        targets = torch.softmax(
            baseline_policy.masked_actor_logits(observations),
            dim=-1,
        ).numpy()
    regularized_batch.regularization_observations = (
        regularized_batch.observations.copy()
    )
    regularized_batch.regularization_targets = np.asarray(
        targets,
        dtype=np.float32,
    )
    regularized_batch.regularization_weights = np.ones(
        len(targets),
        dtype=np.float32,
    )
    regularizer = PolicyProgramRegularizer(
        lambda_extract=0.0,
        lambda_complexity=0.001,
    )

    baseline_trainer.update(batch, regularization_weight=0.0)
    regularized_metrics = regularized_trainer.update(
        regularized_batch,
        regularization_weight=0.0,
        program_regularizer=regularizer,
    )

    for baseline_parameter, regularized_parameter in zip(
        baseline_policy.network.actor.parameters(),
        regularized_policy.network.actor.parameters(),
    ):
        assert torch.equal(baseline_parameter, regularized_parameter)
    assert regularized_metrics["program_regularity_loss"] == 0.0


def test_mappo_backpropagates_forward_program_kl_when_lambda_is_positive() -> None:
    environment_config = WarehouseConfig(horizon=3)
    algorithm_config = MAPPOConfig(
        hidden_dim=16,
        update_epochs=1,
        minibatch_size=10_000,
        seed=1702,
    )
    baseline_policy = MAPPOPolicy(environment_config, algorithm_config)
    regularized_policy = MAPPOPolicy(environment_config, algorithm_config)
    regularized_policy.network.load_state_dict(
        baseline_policy.network.state_dict()
    )
    baseline_trainer = MAPPOTrainer(baseline_policy)
    regularized_trainer = MAPPOTrainer(regularized_policy)
    batch = baseline_trainer.collect_episode(
        WarehouseMultiAgentEnv(environment_config),
        seed=1702,
    )
    regularized_batch = deepcopy(batch)
    with torch.no_grad():
        observations = torch.as_tensor(
            batch.observations,
            dtype=torch.float32,
        )
        actor_probabilities = torch.softmax(
            baseline_policy.masked_actor_logits(observations),
            dim=-1,
        )
        targets = torch.roll(actor_probabilities, shifts=1, dims=-1).numpy()
    regularized_batch.regularization_observations = (
        regularized_batch.observations.copy()
    )
    regularized_batch.regularization_targets = targets
    regularized_batch.regularization_weights = np.ones(
        len(targets),
        dtype=np.float32,
    )
    regularizer = PolicyProgramRegularizer(
        lambda_extract=0.01,
        lambda_complexity=0.001,
    )
    regularizer.program = {"depth": 1, "leaves": 2, "predicates": 1}

    baseline_trainer.update(batch, regularization_weight=0.0)
    regularized_metrics = regularized_trainer.update(
        regularized_batch,
        regularization_weight=0.01,
        program_regularizer=regularizer,
    )

    assert regularized_metrics["program_regularity_loss"] > 0.0
    assert regularized_metrics["program_regularization_total_loss"] > 0.0
    assert any(
        not torch.equal(baseline_parameter, regularized_parameter)
        for baseline_parameter, regularized_parameter in zip(
            baseline_policy.network.actor.parameters(),
            regularized_policy.network.actor.parameters(),
        )
    )


def test_regularization_loss_uses_forward_kl_and_complexity_weights() -> None:
    regularizer = PolicyProgramRegularizer(
        lambda_extract=0.01,
        lambda_complexity=0.001,
        max_depth=4,
        max_leaf_count=6,
        max_predicate_count=4,
    )
    regularizer.program = _program()
    logits = torch.tensor([[2.0, 0.0]], requires_grad=True)
    targets = torch.tensor([[0.7, 0.3]])
    total = regularizer.regularization_loss(
        logits,
        regularizer.program,
        RegularizationStateBatch(targets, torch.ones(1)),
    )

    assert regularizer.last_fidelity_loss is not None
    assert regularizer.last_complexity is not None
    expected = (
        0.01 * regularizer.last_fidelity_loss
        + 0.001 * regularizer.last_complexity.loss
    )
    assert total.item() == pytest.approx(expected)
    total.backward()
    assert logits.grad is not None


def test_warehouse_training_has_no_program_feedback_cli() -> None:
    destinations = {
        action.dest
        for action in train_module.build_parser()._actions
    }

    assert "program_regularization_mode" not in destinations
    assert "use_rcpd" in destinations


def test_program_regularization_json_contains_required_research_metrics() -> None:
    args = train_module.build_parser().parse_args(["--use-rcpd"])
    metrics = SimpleNamespace(
        action_fidelity=0.91,
        mean_kl_divergence=0.08,
        program_depth=4,
        program_size=19,
        program_leaf_count=10,
        program_predicate_count=7,
    )
    rcpd = SimpleNamespace(last_result=SimpleNamespace(metrics=metrics))

    summary = train_module._program_regularization_summary(
        args,
        {"neural_policy": {"mean_reward": 245.3}},
        rcpd,
    )

    assert summary["mode"] == "posthoc_extraction"
    assert summary["runtime_controller"] == (
        "mappo_autoregressive_actor_direct_execution"
    )
    assert summary["lambda_extract"] == 0.0
    assert summary["complexity_lambda"] == pytest.approx(0.001)
    assert summary["extraction_interval"] == 500
    assert summary["program_target_temperature"] == 1.0
    assert summary["minimum_counterfactual_pairs"] == 100
    assert summary["feedback_target"] == "none_posthoc_only"
    assert summary["reward"] == 245.3
    assert summary["action_fidelity"] == 0.91
    assert summary["mean_KL"] == 0.08
    assert summary["program_depth"] == 4
    assert summary["program_size"] == 19
    assert summary["program_leaf_count"] == 10
    assert summary["program_predicate_count"] == 7


def test_training_rcpd_is_structurally_posthoc() -> None:
    args = train_module.build_parser().parse_args(["--use-rcpd"])

    rcpd = train_module._rcpd_from_args(args)

    assert rcpd is not None
    assert rcpd.config.regularization_lambda == 0.0
    assert rcpd.config.maximum_regularization_lambda == 0.0
