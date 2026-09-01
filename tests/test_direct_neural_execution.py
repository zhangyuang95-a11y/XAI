from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from backend.adapters.warehouse import WarehouseAdapter
from env.warehouse.contracts import ACTION_EXECUTION_VERSION, RUNTIME_CONTROLLER
from env.warehouse.environment import WarehouseConfig, WarehouseMultiAgentEnv
from env.warehouse.policy import MAPPOConfig, MAPPOPolicy
from env.warehouse.mappo import MAPPOTrainer, _evaluation_summary
from env.warehouse.joint_risk_loss import (
    expected_collision_loss,
    trainable_joint_pairs,
)
from env.warehouse.scenarios import apply_head_on_scenario
from backend.training import warehouse as train_module


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_production_action_paths_use_causal_runtime_not_legacy_shields() -> None:
    action_paths = (
        "backend/adapters/warehouse.py",
        "ui/web_session.py",
        "ui/development_preview_server.py",
        "env/warehouse/mappo.py",
        "env/warehouse/seed_calibration.py",
        "ui/tutorial.py",
        "evaluate_rl.py",
    )
    forbidden_everywhere = (
        "apply_coordination_shield",
        "coordination_shield_enabled",
        'metadata["coordination_shield"]',
        "stable_coordination_actions",
        "_safe_navigation_teacher_actions",
        "fixed_actions=",
    )
    for relative_path in action_paths:
        source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        for token in forbidden_everywhere:
            assert token not in source, f"{token} remains in {relative_path}"
        if relative_path != "env/warehouse/mappo.py":
            assert "program_regularizer=" not in source
    for relative_path, selector in (
        ("backend/adapters/warehouse.py", "select_ai_ai_joint_actions"),
        ("env/warehouse/seed_calibration.py", "select_ai_ai_joint_actions"),
        ("ui/web_session.py", "select_human_ai_action"),
        ("ui/development_preview_server.py", "select_human_ai_action"),
    ):
        source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        assert selector in source


def test_deterministic_ai_ai_rollout_preserves_actor_proposal_and_submits_joint_choice() -> None:
    config = WarehouseConfig(horizon=1)
    policy = MAPPOPolicy(
        config,
        MAPPOConfig(hidden_dim=16, update_epochs=1, minibatch_size=8, seed=91),
    )
    actor_actions = {"robot_1": "WAIT", "robot_2": "WAIT"}
    original_act = policy.act

    def scripted_act(*args: object, **kwargs: object):
        _, distributions = original_act(*args, **kwargs)
        return dict(actor_actions), distributions

    policy.act = scripted_act  # type: ignore[method-assign]
    environment = WarehouseMultiAgentEnv(config)
    environment.reset(seed=91)
    rollout = WarehouseAdapter(environment).rollout(
        policy,
        horizon=1,
        deterministic=True,
    )

    frame = rollout.frames[0]
    assert frame.proposed_actions == actor_actions
    assert frame.actions == frame.snapshot.metadata["submitted_actions"]
    assert frame.actions != actor_actions
    trace = frame.info["decision_trace"]
    assert trace["policy_actions"] == actor_actions
    assert trace["selected_actions"] == frame.actions
    assert trace["same_frozen_state_for_all_agents"] is True
    assert frame.snapshot.metadata["action_execution"] == (
        "causal_joint_optimizer_atomic_execution"
    )


def test_new_checkpoint_contract_rejects_previous_shield_model() -> None:
    config = WarehouseConfig(horizon=1)
    policy = MAPPOPolicy(config, MAPPOConfig(hidden_dim=16, seed=8))
    payload = {
        "model_version": "warehouse_mappo_v16_open_charger_approach_energy2",
        "environment_version": "warehouse_collaborative_delivery_v9_open_charger_approach",
        "reward_version": "warehouse_safe_mission_reward_v8_open_charger_approach",
        "coordination_shield_enabled": True,
        "environment_config": {},
        "algorithm_config": {},
        "network_state_dict": policy.network.state_dict(),
    }
    with pytest.raises(ValueError, match="Unsupported MAPPO checkpoint version"):
        MAPPOPolicy.from_payload(payload)


def test_wait_memory_contract_rejects_previous_direct_neural_model() -> None:
    config = WarehouseConfig(horizon=1)
    policy = MAPPOPolicy(config, MAPPOConfig(hidden_dim=16, seed=8))
    payload = {
        "model_version": "warehouse_mappo_v25_balanced_autoregressive_neural_energy2",
        "environment_version": "warehouse_collaborative_delivery_v14_balanced_neural_replay",
        "reward_version": "warehouse_safe_mission_reward_v13_user_score_invariant",
        "environment_config": {},
        "algorithm_config": {},
        "network_state_dict": policy.network.state_dict(),
        "action_execution_version": "autoregressive_direct_mappo_actor_action_v3",
        "runtime_controller": "mappo_autoregressive_actor_direct_execution",
    }
    with pytest.raises(ValueError, match="Unsupported MAPPO checkpoint version"):
        MAPPOPolicy.from_payload(payload)


def test_execution_contract_is_explicitly_causal_joint_runtime() -> None:
    assert ACTION_EXECUTION_VERSION == (
        "causal_joint_optimizer_atomic_v16"
    )
    assert RUNTIME_CONTROLLER == (
        "mappo_actor_causal_joint_runtime_v2"
    )


def test_batched_actor_rejects_an_invalid_action_mask_before_device_forward() -> None:
    config = WarehouseConfig(horizon=1)
    policy = MAPPOPolicy(config, MAPPOConfig(hidden_dim=16, seed=92))
    environment = WarehouseMultiAgentEnv(config)
    observations, _ = environment.reset(seed=92)
    invalid = {
        agent_id: np.asarray(observation, dtype=np.float32).copy()
        for agent_id, observation in observations.items()
    }
    invalid["robot_2"][-len(policy.action_names) :] = 0.0

    with pytest.raises(ValueError, match="at least one legal action"):
        policy.act(
            invalid,
            environment.global_state(),
            decision_key=(1, 0),
        )


def test_ppo_optimizes_paired_same_state_expected_collision_risk() -> None:
    config = WarehouseConfig(horizon=6)
    policy = MAPPOPolicy(
        config,
        MAPPOConfig(
            hidden_dim=16,
            intent_dim=8,
            update_epochs=1,
            minibatch_size=12,
            seed=93,
        ),
    )
    trainer = MAPPOTrainer(policy)
    batch = trainer.collect_episode(
        WarehouseMultiAgentEnv(config),
        seed=93,
    )
    batch.trainable_mask[:] = 1.0

    metrics = trainer.update(
        batch,
        joint_collision_loss_weight=0.25,
    )

    assert metrics["joint_collision_loss_weight"] == pytest.approx(0.25)
    assert metrics["joint_collision_pair_updates"] > 0.0
    assert metrics["joint_expected_collision_loss"] >= 0.0

    with pytest.raises(ValueError, match="must be non-negative"):
        trainer.update(batch, joint_collision_loss_weight=-0.1)


def test_joint_risk_loss_equals_manual_p1_c_p2_on_frozen_state() -> None:
    config = WarehouseConfig(horizon=6)
    policy = MAPPOPolicy(
        config,
        MAPPOConfig(hidden_dim=16, intent_dim=8, seed=94),
    )
    environment = WarehouseMultiAgentEnv(config)
    environment.reset(seed=94)
    apply_head_on_scenario(environment, reverse=True, variant=1)
    observations = environment.observations()
    rows = np.stack(
        [policy.actor_input(observations[agent_id]) for agent_id in environment.agent_ids]
    )
    batch = SimpleNamespace(
        observations=rows,
        agent_indices=np.asarray((0, 1), dtype=np.int64),
        trainable_mask=np.ones(2, dtype=np.float32),
    )

    measured = expected_collision_loss(
        policy,
        batch,
        trainable_joint_pairs(batch),
        selected_row_count=2,
        rng=np.random.default_rng(94),
    )
    with torch.no_grad():
        logits = policy.masked_actor_logits(
            torch.as_tensor(rows, dtype=torch.float32)
        )
        probabilities = torch.softmax(logits, dim=-1)
        start = policy.network.joint_collision_matrix_start
        matrix = torch.as_tensor(
            rows[0, start : start + 25].reshape(5, 5),
            dtype=torch.float32,
        )
        expected = probabilities[0] @ matrix @ probabilities[1]

    assert measured.item() == pytest.approx(expected.item())
    # The frozen joint-plan action mask is a hard semantic constraint, so a
    # detected head-on state can have exactly zero residual collision mass.
    assert measured.item() == pytest.approx(0.0)


def test_actor_exposes_trainable_neural_mission_logits() -> None:
    config = WarehouseConfig(horizon=1)
    policy = MAPPOPolicy(
        config,
        MAPPOConfig(hidden_dim=16, intent_dim=8, seed=81),
    )
    environment = WarehouseMultiAgentEnv(config)
    observations, _ = environment.reset(seed=81)
    actor_input = policy.actor_input(observations["robot_1"])
    action_logits, mission_logits = policy.network.actor_outputs(
        torch.as_tensor(actor_input[None, :], dtype=torch.float32)
    )
    assert action_logits.shape == (1, 5)
    assert mission_logits.shape == (1, 5)
    actor_parameter_ids = {
        id(parameter) for parameter in policy.network.actor_parameters()
    }
    assert all(
        id(parameter) in actor_parameter_ids
        for parameter in policy.network.mission_head.parameters()
    )


def test_explicit_mission_model_rejects_v25_checkpoint() -> None:
    config = WarehouseConfig(horizon=1)
    policy = MAPPOPolicy(config, MAPPOConfig(hidden_dim=16, seed=8))
    payload = {
        "model_version": (
            "warehouse_mappo_v32_direct_structured_actor_commitment"
        ),
        "environment_version": (
            "warehouse_collaborative_delivery_v19_structured_neural_commitment"
        ),
        "reward_version": (
            "warehouse_safe_mission_reward_v16_continuous_commitment"
        ),
        "environment_config": {},
        "algorithm_config": {},
        "network_state_dict": policy.network.state_dict(),
    }
    with pytest.raises(ValueError, match="Unsupported MAPPO checkpoint version"):
        MAPPOPolicy.from_payload(payload)


def test_formal_metrics_expose_repeated_collision_tail_risk() -> None:
    summary = _evaluation_summary(
        training_rewards=[0.0, 0.0],
        base_training_rewards=[0.0, 0.0],
        potential_shaping_rewards=[0.0, 0.0],
        user_scores=[0.0, 0.0],
        deliveries=[0, 0],
        steps=[120, 120],
        collision_episodes=1,
        shutdown_episodes=0,
        collision_counts=[0, 12],
        shutdown_counts=[0, 0],
        charger_use_steps=0,
        detour_units=0.0,
        delivery_durations=[],
        minimum_batteries=[10.0, 10.0],
        terminal_reasons={"horizon": 2},
    )

    assert summary["maximum_robot_collision_events"] == 12
    assert summary["repeated_collision_episode_rate"] == 0.5
    assert summary["collision_event_samples"] == [0, 12]


def test_rcpd_training_source_is_only_executed_neural_rollout_rows() -> None:
    source = (PROJECT_ROOT / "backend/training/warehouse.py").read_text(
        encoding="utf-8"
    )
    assert "_shared_task_probe_records" not in source
    assert '"training_data_source": "executed_neural_rollout_only"' in source
    assert "program_regularizer=" not in source
    assert "def _attach_regularization" not in source
    assert "def _records_from_batches" not in source
    assert "program_regularization_mode" not in source
    assert "regularization_weight=" not in source
    assert "--rcpd-feedback" not in source
    assert "--rcpd-lambda" not in source
    train_body = source[source.index("def train(") :]
    assert train_body.index("if args.use_rcpd:") < train_body.index(
        "rcpd = _rcpd_from_args(args)"
    )
    assert "rcpd = _rcpd_from_args(args)" not in train_body[
        : train_body.index("if args.use_rcpd:")
    ]


def test_reference_and_seed_artifacts_declare_causal_runtime_provenance() -> None:
    tutorial = (PROJECT_ROOT / "ui/tutorial.py").read_text(encoding="utf-8")
    seed_calibration = (
        PROJECT_ROOT / "env/warehouse/seed_calibration.py"
    ).read_text(encoding="utf-8")
    artifact_contracts = (
        PROJECT_ROOT / "backend/artifact_contracts.py"
    ).read_text(encoding="utf-8")

    assert '"rollout_action_source": RUNTIME_ACTION_SOURCE' in tutorial
    assert '"post_policy_action_interventions": int(runtime_overrides)' in tutorial
    assert '"runtime_controller": RUNTIME_CONTROLLER' in tutorial
    assert '"rollout_action_source": RUNTIME_ACTION_SOURCE' in seed_calibration
    assert '"post_policy_action_interventions": sum(' in seed_calibration
    assert '"runtime_controller": RUNTIME_CONTROLLER' in seed_calibration
    assert 'payload.get("rollout_action_source")' in artifact_contracts
    assert 'payload.get("post_policy_action_interventions", -1)' in (
        artifact_contracts
    )


def test_web_runtime_rejects_rcpd_programs_that_could_control_actions() -> None:
    source = (PROJECT_ROOT / "backend/artifact_contracts.py").read_text(
        encoding="utf-8"
    )
    assert 'metadata.get("distilled_component") != "neural_actor"' in source
    assert 'metadata.get("training_data_source")' in source
    assert '!= "executed_neural_rollout_only"' in source
    assert 'metadata.get("runtime_controller") != RUNTIME_CONTROLLER' in source
    assert 'bool(metadata.get("runtime_control_allowed", True))' in source
    assert 'bool(metadata.get("feedback_allowed", True))' in source
    assert 'metadata.get("training_role") != "posthoc_explanation_only"' in source
    assert '!= ("local_explanation_audit",)' in source
    assert 'bool(metadata.get("regularization_version", True))' in source


def test_formal_evaluation_requires_disjoint_declared_training_seed_ledger() -> None:
    training = (PROJECT_ROOT / "backend/training/warehouse.py").read_text(
        encoding="utf-8"
    )
    ledger_source = (
        PROJECT_ROOT / "backend/training/seed_ledger.py"
    ).read_text(encoding="utf-8")
    evaluation = (PROJECT_ROOT / "evaluate_rl.py").read_text(encoding="utf-8")
    assert '"schema": "warehouse-training-seed-ledger.v1"' in ledger_source
    assert '"seed_ledger": training_seed_ledger(args)' in training
    assert (
        'ledger.get("schema") == "warehouse-training-seed-ledger.v1"'
        in evaluation
    )
    assert "for reserved_range in reserved_ranges" in evaluation
    assert "for formal_range in formal_exclusive_ranges" in evaluation


def test_posthoc_rcpd_fit_emits_the_direct_actor_explanation_only_contract() -> None:
    captured: dict[str, object] = {}

    class FakeRCPD:
        def maybe_extract(self, *args: object, **kwargs: object) -> object:
            del args
            captured.update(kwargs)
            return object()

    class FakeAdapter:
        @staticmethod
        def action_legality_features() -> dict[str, str]:
            return {}

        @staticmethod
        def action_constraint_reason_features() -> dict[str, dict[str, str]]:
            return {}

        @staticmethod
        def required_program_predicate_groups() -> dict[str, tuple[str, ...]]:
            return {}

    assert train_module._fit_rcpd(  # type: ignore[arg-type]
        FakeRCPD(),
        [{"probabilities": {}, "features": {}}],
        FakeAdapter(),
        step=1,
    )
    metadata = captured["program_metadata"]
    assert isinstance(metadata, dict)
    assert metadata["action_execution_version"] == ACTION_EXECUTION_VERSION
    assert metadata["runtime_controller"] == RUNTIME_CONTROLLER
    assert metadata["runtime_control_allowed"] is False
    assert metadata["feedback_allowed"] is False
    assert metadata["training_role"] == "posthoc_explanation_only"
    assert metadata["program_roles"] == ("local_explanation_audit",)
    assert metadata["regularization_version"] is False
