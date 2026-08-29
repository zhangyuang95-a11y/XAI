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
from env.warehouse import scenario_evaluation as scenario_evaluation_module
from env.warehouse.observations import _coordination_features, observation_dim
from env.warehouse.navigation import ACTIONS
from env.warehouse.layouts import STAGGERED_AISLES_LAYOUT
from env.warehouse.contracts import RUNTIME_CONTROLLER
from env.warehouse.scenarios import (
    apply_charger_handoff_scenario,
    apply_critical_charger_approach_scenario,
    apply_delivery_goal_clearance_scenario,
    apply_head_on_scenario,
    apply_same_target_conflict_scenario,
)
from env.warehouse.coordination import stable_coordination_actions
from env.warehouse.energy_management import (
    charge_release_energy,
    charger_service_required,
)


def test_goal_clearance_curriculum_labels_two_step_follow_through() -> None:
    config = WarehouseConfig(horizon=8)
    environment = WarehouseMultiAgentEnv(config)
    environment.reset(seed=819)
    apply_delivery_goal_clearance_scenario(environment, variant=17)
    state = environment.get_state()

    labels = train_module._safe_navigation_teacher_actions(environment)
    targets = environment._resolve_motion(state, labels)[0]

    assert targets["robot_1"] == state.by_id("robot_1").position
    assert targets["robot_2"] != state.by_id("robot_2").position

    environment.step(labels)
    next_state = environment.get_state()
    next_labels = train_module._safe_navigation_teacher_actions(environment)
    next_targets = environment._resolve_motion(next_state, next_labels)[0]
    assert (
        next_targets["robot_1"]
        == state.by_id("robot_1").navigation_goal_position
    )
    assert "junction_conflict" in train_module.STRONG_ACTOR_CORRECTION_CATEGORIES
from env.warehouse.policy import (
    independent_actor_input,
)
from backend.training import warehouse as train_module
from backend.training import learner_dataset as learner_dataset_module
from backend.training import partner_risk as partner_risk_module
from backend.training.learner_replay import (
    MAXIMUM_OVERSAMPLE_FACTOR,
    REPLAY_CATEGORIES,
    fit_actor_supervised,
)
from backend.training.learner_dataset import best_unilateral_mission_action


def test_partner_risk_all_scope_updates_participant_forecast_parameters() -> None:
    config = WarehouseConfig(horizon=8)
    policy = MAPPOPolicy(
        config,
        MAPPOConfig(hidden_dim=16, intent_dim=8, seed=2_621),
    )
    selected_ids = {
        id(parameter)
        for parameter in partner_risk_module._optimization_parameters(
            policy,
            "all",
        )
    }
    participant_forecast_ids = {
        id(parameter)
        for parameter in policy.network.participant_context_predictor.parameters()
    }

    assert participant_forecast_ids
    assert participant_forecast_ids.issubset(selected_ids)
    assert participant_forecast_ids.isdisjoint(
        {id(parameter) for parameter in policy.network.ppo_actor_parameters()}
    )


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
        observation_dim(config),
    )
    assert labels.shape == (64,)
    assert categories.shape == (64,)
    assert set(categories).issubset(
        set(REPLAY_CATEGORIES)
    )
    assert coverage["head_on_rows"] > 0
    assert coverage["counterfactual_teammate_rows"] == 0


def test_learner_relabel_can_return_same_state_teammate_labels() -> None:
    config = WarehouseConfig(horizon=12)
    policy = MAPPOPolicy(
        config,
        MAPPOConfig(hidden_dim=16, update_epochs=1, minibatch_size=32, seed=311),
    )

    rows, labels, teammate_labels, categories, coverage = (
        train_module._collect_learner_state_relabel_dataset(
            policy,
            config,
            sample_count=48,
            seed=18_811,
            include_teammate_labels=True,
        )
    )

    assert rows.shape == (48, observation_dim(config))
    assert labels.shape == teammate_labels.shape == categories.shape == (48,)
    assert np.any(teammate_labels >= 0)
    assert np.all(teammate_labels < len(ACTIONS))
    assert coverage["ordinary_rows"] > 0


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
    assert set(categories).issubset({"joint_wait", "charger_queue"})
    assert "joint_wait" in set(categories)


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
        observation_dim(config),
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
    original_reset(seed=914)
    apply_same_target_conflict_scenario(environment, variant=0)
    conflict_state = environment.get_state()
    actor_collision = next(
        {"robot_1": first, "robot_2": second}
        for first in ACTIONS
        for second in ACTIONS
        if environment._resolve_motion(
            conflict_state,
            {"robot_1": first, "robot_2": second},
        )[3]
    )
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
        observation_dim(config),
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
        observation_dim(config),
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
        observation_dim(config),
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


def test_unilateral_mission_correction_never_enters_peer_frozen_cell() -> None:
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
        agent_id="robot_1",
    )
    assert correction == "WAIT"


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


def test_lower_battery_waiter_receives_two_phase_charger_handoff() -> None:
    environment = WarehouseMultiAgentEnv(
        WarehouseConfig(horizon=8, participant_detour_scoring=False)
    )
    environment.reset(seed=940)
    apply_charger_handoff_scenario(
        environment,
        occupant_agent_id="robot_2",
        occupant_battery=58.0,
        queued_battery=36.0,
        occupant_carrying=True,
        queued_carrying=True,
    )

    before = environment.get_state()
    occupant_priority = _coordination_features(
        before,
        "robot_2",
        environment.config,
    )[6:8]
    waiter_priority = _coordination_features(
        before,
        "robot_1",
        environment.config,
    )[6:8]
    teacher = train_module.stable_coordination_actions(environment)
    targets, _, invalid, collision, _, _ = environment._resolve_motion(
        before,
        teacher,
    )

    assert teacher["robot_2"] in {"LEFT", "RIGHT"}
    assert teacher["robot_1"] == "WAIT"
    assert occupant_priority == [0.0, 1.0]
    assert waiter_priority == [1.0, 0.0]
    assert not invalid
    assert not collision
    assert targets["robot_1"] == before.by_id("robot_1").position

    waiting = WarehouseMultiAgentEnv(
        WarehouseConfig(horizon=8, participant_detour_scoring=False)
    )
    waiting.reset(seed=940)
    waiting.set_state(before)
    _, _, _, _, wait_info = waiting.step(
        {"robot_1": "WAIT", "robot_2": "WAIT"}
    )
    _, _, _, _, teacher_info = environment.step(teacher)

    assert wait_info["counterfactual_regret_units"] == {
        "robot_1": 0.0,
        "robot_2": 1.0,
    }
    assert wait_info["avoidable_wait_agents"] == ("robot_2",)
    assert wait_info["joint_wait_escape_actions"] == teacher
    assert teacher_info["counterfactual_regret_units"] == {
        "robot_1": 0.0,
        "robot_2": 0.0,
    }


def test_lower_battery_charger_occupant_retains_station() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=8))
    environment.reset(seed=940)
    apply_charger_handoff_scenario(
        environment,
        occupant_agent_id="robot_2",
        occupant_battery=20.0,
        queued_battery=24.0,
        occupant_carrying=True,
        queued_carrying=True,
    )

    before = environment.get_state()
    occupant_priority = _coordination_features(
        before,
        "robot_2",
        environment.config,
    )[6:8]
    teacher = train_module.stable_coordination_actions(environment)
    targets, _, invalid, collision, _, _ = environment._resolve_motion(
        before,
        teacher,
    )

    assert environment._requires_charge(before, before.by_id("robot_2"))
    assert teacher == {"robot_1": "WAIT", "robot_2": "WAIT"}
    assert occupant_priority == [1.0, 0.0]
    assert not invalid
    assert not collision
    assert targets == {
        agent.agent_id: agent.position
        for agent in before.agents
    }


def test_small_charger_energy_gap_does_not_create_one_wait_ping_pong() -> None:
    environment = WarehouseMultiAgentEnv(WarehouseConfig(horizon=120))
    environment.reset(seed=940)
    apply_charger_handoff_scenario(
        environment,
        occupant_agent_id="robot_2",
        occupant_battery=38.0,
        queued_battery=30.0,
        occupant_carrying=True,
        queued_carrying=True,
    )

    state = environment.get_state()
    teacher = stable_coordination_actions(environment)
    occupant_priority = _coordination_features(
        state,
        "robot_2",
        environment.config,
    )[6:8]

    assert teacher == {"robot_1": "WAIT", "robot_2": "WAIT"}
    assert occupant_priority == [1.0, 0.0]


def test_hysteretic_charging_wait_is_not_relabelled_as_joint_escape() -> None:
    environment = WarehouseMultiAgentEnv(
        WarehouseConfig(horizon=120, participant_detour_scoring=False)
    )
    environment.reset(seed=1)
    state = environment.get_state()
    charger_row, charger_column = environment.layout.charger_position
    occupant = state.by_id("robot_2")
    waiter = state.by_id("robot_1")
    occupant.position = environment.layout.charger_position
    occupant.battery = 50.0
    occupant.charge_mode_active = True
    occupant.navigation_goal_kind = "charge"
    occupant.navigation_goal_position = environment.layout.charger_position
    waiter.position = (charger_row, charger_column + 1)
    waiter.battery = 46.0
    waiter.charge_mode_active = True
    waiter.navigation_goal_kind = "charge"
    waiter.navigation_goal_position = environment.layout.charger_position
    environment.set_state(state)
    before = environment.get_state()
    occupant = before.by_id("robot_2")

    assert not environment._requires_charge(before, occupant)
    assert occupant.battery < charge_release_energy(
        environment,
        before,
        occupant,
    )
    assert charger_service_required(environment, before, occupant)
    assert stable_coordination_actions(environment) == {
        "robot_1": "WAIT",
        "robot_2": "WAIT",
    }

    _, _, _, _, info = environment.step(
        {"robot_1": "WAIT", "robot_2": "WAIT"}
    )

    assert info["counterfactual_regret_units"] == {
        "robot_1": 0.0,
        "robot_2": 0.0,
    }
    assert info["avoidable_wait_agents"] == ()
    assert info["joint_wait_escape_actions"] == {}


def test_actor_leaves_charger_after_charge_goal_releases() -> None:
    config = WarehouseConfig(horizon=120)
    environment = WarehouseMultiAgentEnv(config)
    environment.reset(seed=941)
    state = environment.get_state()
    robot = state.by_id("robot_1")
    task = state.tasks[0]
    robot.position = environment.layout.charger_position
    robot.battery = 80.0
    robot.charge_mode_active = False
    robot.carrying_task_id = task.task_id
    task.status = "carried"
    task.carrier_agent_id = robot.agent_id
    task.delivery_position = (5, 4)
    environment.set_state(state)

    before = environment.get_state()
    assert before.by_id(robot.agent_id).navigation_goal_kind == "delivery"
    assert not charger_service_required(
        environment,
        before,
        before.by_id(robot.agent_id),
    )
    policy = MAPPOPolicy(
        config,
        MAPPOConfig(hidden_dim=16, intent_dim=8, seed=942),
    )

    actions, distributions = policy.act(
        environment.observations(),
        environment.global_state(),
        deterministic=True,
    )

    assert actions[robot.agent_id] != "WAIT"
    assert (
        distributions[robot.agent_id].probabilities[ACTIONS.index("WAIT")]
        < 1e-6
    )


def test_actor_enters_empty_charger_after_productive_causal_handoff() -> None:
    config = WarehouseConfig(horizon=120)
    environment = WarehouseMultiAgentEnv(config)
    environment.reset(seed=942)
    state = environment.get_state()
    state.frame = 46
    charger_row, charger_column = environment.layout.charger_position
    returning = state.by_id("robot_1")
    teammate = state.by_id("robot_2")
    returning.position = (charger_row, charger_column + 1)
    returning.battery = 30.0
    returning.navigation_goal_kind = "charge"
    returning.navigation_goal_position = environment.layout.charger_position
    returning.charge_mode_active = True
    returning.last_charger_departure_frame = 42
    returning.team_deliveries_at_last_charger_departure = state.total_deliveries
    teammate.position = (charger_row, charger_column - 1)
    teammate.battery = 46.0
    teammate.last_action = "LEFT"
    teammate.last_executed_action = "LEFT"
    teammate.last_battery_delta = -config.move_battery_cost
    teammate.steps_since_charging = 1
    teammate.last_charger_departure_frame = state.frame
    teammate.navigation_goal_kind = "wait"
    teammate.navigation_goal_position = teammate.position
    environment.set_state(state)
    observations = environment.observations()
    policy = MAPPOPolicy(
        config,
        MAPPOConfig(hidden_dim=16, intent_dim=8, seed=943),
    )

    actions, distributions = policy.act(
        observations,
        environment.global_state(),
        deterministic=True,
    )
    entry = "LEFT"

    assert stable_coordination_actions(environment)["robot_1"] == entry
    assert actions["robot_1"] == entry
    assert (
        distributions["robot_1"].probabilities[ACTIONS.index(entry)]
        > distributions["robot_1"].probabilities[ACTIONS.index("WAIT")]
    )


def test_actor_does_not_immediately_reverse_into_charger_without_progress() -> None:
    config = WarehouseConfig(horizon=120)
    environment = WarehouseMultiAgentEnv(config)
    environment.reset(seed=944)
    state = environment.get_state()
    state.frame = 109
    charger_row, charger_column = environment.layout.charger_position
    returning = state.by_id("robot_1")
    teammate = state.by_id("robot_2")
    returning.position = (charger_row - 1, charger_column)
    returning.battery = 70.0
    returning.carrying_task_id = None
    returning.navigation_goal_kind = "wait"
    returning.navigation_goal_position = returning.position
    returning.last_charger_departure_frame = state.frame
    returning.team_deliveries_at_last_charger_departure = state.total_deliveries
    teammate.position = (charger_row - 2, charger_column - 1)
    teammate.steps_since_charging = 8
    environment.set_state(state)
    policy = MAPPOPolicy(
        config,
        MAPPOConfig(hidden_dim=16, intent_dim=8, seed=945),
    )

    actions, distributions = policy.act(
        environment.observations(),
        environment.global_state(),
        deterministic=True,
    )
    reverse = "DOWN"

    assert actions["robot_1"] != reverse
    assert (
        distributions["robot_1"].probabilities[ACTIONS.index(reverse)]
        < distributions["robot_1"].probabilities[ACTIONS.index("WAIT")]
    )


def test_offline_teacher_clears_a_delivery_goal_without_blocking_the_loaded_robot() -> None:
    legacy_config = WarehouseConfig(
        rows=STAGGERED_AISLES_LAYOUT.rows,
        cols=STAGGERED_AISLES_LAYOUT.cols,
        map_layout_id=STAGGERED_AISLES_LAYOUT.layout_id,
    )
    environment = WarehouseMultiAgentEnv(legacy_config)
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

    # Robot 2 cannot assume that Robot 1 will vacate or avoid its next route
    # cell in the same frame.  Clear Robot 1 first, then advance from the next
    # frozen state; the old simultaneous exact-action assertion encoded a
    # current-frame action dependency.
    assert teacher["robot_1"] != "WAIT"
    assert teacher["robot_2"] == "WAIT"
    assert targets["robot_1"] not in {
        before.by_id("robot_2").navigation_goal_position,
        (3, 5),
    }
    environment.step(teacher)
    cleared = environment.get_state()
    follow = train_module.stable_coordination_actions(environment)
    follow_targets = environment._resolve_motion(cleared, follow)[0]
    assert follow["robot_2"] != "WAIT"
    assert shortest_path_distance(
        follow_targets["robot_2"],
        cleared.by_id("robot_2").navigation_goal_position,
        legacy_config.map_layout_id,
    ) < shortest_path_distance(
        cleared.by_id("robot_2").position,
        cleared.by_id("robot_2").navigation_goal_position,
        legacy_config.map_layout_id,
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
    row = independent_actor_input(observations["robot_1"])
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
    row = independent_actor_input(observations["robot_1"])
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
    row = independent_actor_input(observations["robot_1"])
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
            independent_actor_input(observations["robot_1"]),
            independent_actor_input(observations["robot_2"]),
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


def test_ppo_update_preserves_supervised_peer_forecast_parameters() -> None:
    config = WarehouseConfig(horizon=6)
    policy = MAPPOPolicy(
        config,
        MAPPOConfig(hidden_dim=16, update_epochs=1, minibatch_size=32, seed=235),
    )
    trainer = MAPPOTrainer(policy)
    environment = WarehouseMultiAgentEnv(config)
    batch = trainer.collect_episode(environment, seed=2916)
    forecast_before = tuple(
        parameter.detach().clone()
        for module in (
            policy.network.teammate_action_predictor,
            policy.network.teammate_context_predictor,
            policy.network.participant_context_predictor,
        )
        for parameter in module.parameters()
    )

    trainer.update(batch)

    forecast_after = tuple(
        parameter.detach()
        for module in (
            policy.network.teammate_action_predictor,
            policy.network.teammate_context_predictor,
            policy.network.participant_context_predictor,
        )
        for parameter in module.parameters()
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(forecast_before, forecast_after)
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
    row = independent_actor_input(observations["robot_1"])
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
    row = independent_actor_input(observations["robot_1"])
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
    row = independent_actor_input(observations["robot_1"])
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


def test_formal_compact_coordination_scenarios_measure_completed_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = WarehouseConfig(horizon=120)
    active_environments: list[WarehouseMultiAgentEnv] = []
    real_environment = WarehouseMultiAgentEnv

    class TrackingEnvironment(real_environment):
        def __init__(self, environment_config: WarehouseConfig) -> None:
            super().__init__(environment_config)
            active_environments.append(self)

    class FrozenStateTeacherInference:
        def act(self, *_args, **_kwargs):
            return stable_coordination_actions(active_environments[-1]), {}

    class FrozenStateTeacherPolicy:
        environment_config = config

        @staticmethod
        def fork_for_inference(*_args, **_kwargs):
            return FrozenStateTeacherInference()

    monkeypatch.setattr(
        scenario_evaluation_module,
        "WarehouseMultiAgentEnv",
        TrackingEnvironment,
    )
    policy = FrozenStateTeacherPolicy()
    empty = scenario_evaluation_module.evaluate_empty_delivery_clearance_scenarios(
        policy,  # type: ignore[arg-type]
        config,
        episodes=16,
        seed=4_500,
    )
    dual = scenario_evaluation_module.evaluate_dual_charger_approach_scenarios(
        policy,  # type: ignore[arg-type]
        config,
        episodes=16,
        seed=4_600,
    )
    outer = scenario_evaluation_module.evaluate_outer_exit_charger_approach_scenarios(
        policy,  # type: ignore[arg-type]
        config,
        episodes=48,
        seed=4_700,
    )
    occupied = scenario_evaluation_module.evaluate_occupied_charger_handoff_scenarios(
        policy,  # type: ignore[arg-type]
        config,
        episodes=32,
        seed=4_800,
    )

    assert empty["success_rate"] == 1.0
    assert empty["collision_rate"] == 0.0
    assert empty["mean_completion_steps"] == 2.0
    assert dual["success_rate"] == 1.0
    assert dual["collision_rate"] == 0.0
    assert dual["return_cycles_per_episode"] == 0.0
    assert outer["success_rate"] == 1.0
    assert outer["collision_rate"] == 0.0
    assert outer["mean_completion_steps"] == 3.0
    assert occupied["success_rate"] == 1.0
    assert occupied["collision_rate"] == 0.0
    assert occupied["priority_violation_rate"] == 0.0


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
    assert summary["runtime_controller"] == RUNTIME_CONTROLLER
    assert summary["lambda_extract"] == 0.0
    assert summary["complexity_lambda"] == pytest.approx(0.001)
    assert summary["extraction_interval"] == 500
    assert summary["program_target_temperature"] == 1.5
    assert summary["action_structure_weight"] == pytest.approx(0.25)
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
