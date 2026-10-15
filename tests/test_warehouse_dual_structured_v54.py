"""Small integration checks for the isolated structured Warehouse trainer."""
from __future__ import annotations

from copy import deepcopy
import json
import sys

import numpy as np
import pytest
import torch

from backend.training import warehouse_dual_structured_v54 as trainer
from backend.training import run_warehouse_structured_v54_variant as runner
from backend.training.warehouse_structured_v5 import (
    FrozenStructuredDualNN, STRUCTURED_FEATURE_SCHEMA_SHA256,
    STRUCTURED_MODEL_VERSION,
)
from env.warehouse.domain import participant_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import AGENTS, action_masks, observations


def _public_observation():
    env = WarehouseMultiAgentEnv(participant_study_config())
    env.reset(seed=1000)
    return observations(env.state), action_masks(env.state)


def test_structured_trainer_has_isolated_artifacts_and_seed_blocks(tmp_path):
    assert trainer.DEFAULT_SEEDS == (6101, 6102, 6103)
    assert trainer.DEVELOPMENT_PILOT_SEED == 6099
    assert trainer.FULL_JOINT_STEPS == 200_000
    assert trainer.PROGRAM_LAMBDA == 0.01
    assert trainer._program_coefficient(0.8) == 0.0
    assert trainer._program_coefficient(0.85) == pytest.approx(0.01)
    assert trainer.STRUCTURED_CONFIG != trainer.CONFIG_TEMPLATE
    assert trainer.STRUCTURED_CONFIG.name.endswith("structured_v54.json")
    blocks = [set(trainer.PILOT_DEVELOPMENT_SEEDS),
              set(trainer.DEVELOPMENT_SEEDS),
              set(trainer.PRELIMINARY_PROGRAM_AUDIT_SEEDS),
              set(trainer.FINAL_PROGRAM_AUDIT_SEEDS),
              {seed for task_seeds in trainer.HELDOUT_SEEDS_BY_TASK.values()
               for seed in task_seeds}]
    assert all(not left & right for i, left in enumerate(blocks)
               for right in blocks[i + 1:])
    with pytest.raises(ValueError, match="development pilot cannot"):
        trainer.run_experiment(pair_seeds=(6099,), output_dir=tmp_path,
                               development_pilot=True, enable_heldout=True)
    with pytest.raises(ValueError, match="must use seed 6099"):
        trainer.run_experiment(pair_seeds=(6101,), output_dir=tmp_path,
                               development_pilot=True)
    with pytest.raises(ValueError, match="must use seeds 6101"):
        trainer.run_experiment(pair_seeds=(6101, 6102, 6104), output_dir=tmp_path)
    assert not list(tmp_path.iterdir())


def test_structured_bc_interface_and_paired_role_hashes(monkeypatch):
    # One offline teacher episode and one NN-visited episode exercise the BC
    # interface without opening any formal experiment or reserved seed block.
    monkeypatch.setattr(trainer, "DAGGER_ROUNDS", 1)
    monkeypatch.setattr(trainer, "DAGGER_ORIGINAL_TEACHER_EPISODES", 1)
    monkeypatch.setattr(trainer, "DAGGER_VISITED_EPISODES_PER_ROUND", 1)
    monkeypatch.setattr(trainer, "DAGGER_EPOCHS_PER_ROUND", 1)
    torch.manual_seed(10123)
    model = trainer.DualNN()
    initial = trainer._weight_hashes(model)
    report = trainer.pretrain_pair(model, seed=10123, episodes=1, epochs=1)
    assert report["initial_role_sha256"] == initial
    assert report["pretrained_role_sha256"] == trainer._weight_hashes(model)
    assert all(report["pretrained_role_sha256"][role] != initial[role]
               for role in AGENTS)
    assert report["offline_nn_visit_actions_from_actor_only"]
    assert len(report["dagger_rounds"]) == 1


def test_structured_ppo_and_sparse_oracle_use_same_masked_actor():
    torch.manual_seed(73)
    model = trainer.DualNN()
    obs, masks = _public_observation()
    with torch.no_grad():
        logits = model.outputs(torch.from_numpy(obs)[None],
                               torch.from_numpy(masks)[None])[0]
        probabilities = torch.softmax(logits, -1)
        actions = probabilities.argmax(-1)
        old_logp = torch.log(probabilities.gather(-1, actions[:, None]).squeeze(-1))
        values = model.critic(torch.from_numpy(obs).flatten()[None])[0]
    for row, role in enumerate(AGENTS):
        oracle = trainer._oracle(model, role)({"obs": obs[row], "mask": masks[row]})
        np.testing.assert_allclose(list(oracle.values()), probabilities[row].numpy(), atol=1e-6)

    episode = {"records": [{"obs": obs, "mask": masks,
                            "action": actions.numpy(), "logp": old_logp.numpy(),
                            "value": values.numpy(),
                            "reward": np.asarray([1.0, 2.0], dtype=np.float32),
                            "seed": 1000, "frame": 0}],
               "bootstrap": np.zeros(2, dtype=np.float32)}

    class CoveredProgram:
        def __init__(self):
            self.calls = 0

        def covered(self, sample, *, actor_action, purpose):
            self.calls += 1
            assert purpose == "feedback"
            assert sample["mask"][ACTIONS.index(actor_action)] > 0.5
            legal = np.flatnonzero(sample["mask"] > 0.5)
            distribution = {action: 0.0 for action in ACTIONS}
            for index in legal:
                distribution[ACTIONS[index]] = 0.1 / len(legal)
            distribution[actor_action] += 0.9
            return {"probabilities": distribution}

    programs = {role: CoveredProgram() for role in AGENTS}
    initial = trainer._weight_hashes(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    update = trainer.ppo_update(model, optimizer, [episode], programs,
                                progress=0.85, epochs=1)
    assert update["feedback_rows"] == {role: 1 for role in AGENTS}
    assert update["zero_action_coverage_rows"] == {role: 0 for role in AGENTS}
    assert all(program.calls == 1 for program in programs.values())
    assert all(trainer._weight_hashes(model)[role] != initial[role] for role in AGENTS)
    raw = trainer.ppo_update(model, optimizer, [deepcopy(episode)], None,
                             progress=0.85, epochs=1)
    assert raw["feedback_rows"] == {role: 0 for role in AGENTS}
    assert raw["feedback_window_actor_rows"] == {role: 0 for role in AGENTS}


def test_short_self_play_export_and_frozen_inference_bind_same_roles(tmp_path):
    torch.manual_seed(81)
    model = trainer.DualNN()
    episode, summary = trainer.collect_episode(
        model, seed=10123, generator=torch.Generator().manual_seed(81), max_steps=3)
    assert summary["joint_steps"] == len(episode["records"]) == 3
    assert summary["actor_decisions"] == 6
    assert summary["post_policy_interventions"] == 0
    actor_path = tmp_path / "warehouse_dual_structured_v54_raw_actor.npz"
    artifact = trainer.export_actor(model, actor_path, pair_seed=6099, variant="raw")
    assert artifact["feature_schema_sha256"] == STRUCTURED_FEATURE_SCHEMA_SHA256
    assert artifact["role_sha256"] == trainer._weight_hashes(model)
    frozen = FrozenStructuredDualNN.load(
        actor_path, expected_sha256=artifact["sha256"],
        expected_role_sha256=artifact["role_sha256"])
    obs, masks = _public_observation()
    with torch.no_grad():
        expected = model.outputs(torch.from_numpy(obs), torch.from_numpy(masks)).numpy()
    actual = np.where(masks > 0.5, frozen.logits(obs, masks), -1.0e9)
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-5)
    assert trainer.STRUCTURED_FEATURE_SCHEMA_SHA256 in trainer.BC_ARCHITECTURE_VERSION
    assert frozen.role_sha256 == artifact["role_sha256"]


def test_structured_variant_runner_rejects_stale_bc_and_unfinished_pilot(tmp_path):
    torch.manual_seed(6099)
    model = trainer.DualNN()
    work = runner._work_dir(tmp_path)
    work.mkdir()
    path = runner._bc_path(tmp_path, 6099)
    payload = {"pair_seed": 6099, "bc_episodes": 200, "bc_epochs": 30,
               "actor_model_version": trainer.STRUCTURED_MODEL_VERSION,
               "feature_schema_sha256": "0" * 64,
               "architecture_version": trainer.BC_ARCHITECTURE_VERSION,
               "state_dict": model.state_dict(),
               "report": {"pretrained_role_sha256": trainer._weight_hashes(model)}}
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="BC cache mismatch"):
        runner._load_bc(path, 6099)
    payload["feature_schema_sha256"] = trainer.STRUCTURED_FEATURE_SCHEMA_SHA256
    torch.save(payload, path)
    assert runner._load_bc(path, 6099)["pair_seed"] == 6099
    with pytest.raises(FileNotFoundError):
        runner._pilot_checkpoints_complete(tmp_path, 6099, 200_000)


def test_structured_completed_checkpoint_rejects_report_actor_hash_drift(tmp_path):
    torch.manual_seed(6099)
    model = trainer.DualNN()
    initial = deepcopy(model.state_dict())
    hashes = trainer._weight_hashes(model)
    checkpoint = tmp_path / "pair_6099_raw_matched.pt"
    torch.save({
        "pair_seed": 6099, "variant": "raw", "joint_steps": 10,
        "actor_model_version": trainer.STRUCTURED_MODEL_VERSION,
        "feature_schema_sha256": trainer.STRUCTURED_FEATURE_SCHEMA_SHA256,
        "scenario_schedule_version": trainer.SCENARIO_SCHEDULE_VERSION,
        "training_reward_version": trainer.TRAINING_REWARD_VERSION,
        "bc_architecture_version": trainer.BC_ARCHITECTURE_VERSION,
        "program_feedback_version": trainer.PROGRAM_FEEDBACK_VERSION,
        "initial_role_sha256": hashes, "complete": True,
        "selected_state_dict": initial, "selected_programs": {},
        "report": {"actor_model_version": trainer.STRUCTURED_MODEL_VERSION,
                   "feature_schema_sha256": trainer.STRUCTURED_FEATURE_SCHEMA_SHA256,
                   "joint_steps": 10, "initial_role_sha256": hashes,
                   "selected_role_sha256": {role: "0" * 64 for role in AGENTS}},
    }, checkpoint)
    with pytest.raises(RuntimeError, match="checkpoint report mismatch"):
        trainer.train_variant(initial, pair_seed=6099, variant="raw",
                              joint_steps=10, checkpoint_path=checkpoint)


def test_structured_pilot_acceptance_feeds_existing_development_preflight(
        tmp_path, monkeypatch, capsys):
    torch.manual_seed(6099)
    model = trainer.DualNN()
    work = runner._work_dir(tmp_path)
    work.mkdir()
    torch.save({"pair_seed": 6099, "bc_episodes": 200, "bc_epochs": 30,
                "actor_model_version": trainer.STRUCTURED_MODEL_VERSION,
                "feature_schema_sha256": trainer.STRUCTURED_FEATURE_SCHEMA_SHA256,
                "architecture_version": trainer.BC_ARCHITECTURE_VERSION,
                "state_dict": model.state_dict(),
                "report": {"pretrained_role_sha256": trainer._weight_hashes(model)}},
               runner._bc_path(tmp_path, 6099))
    for variant in ("raw", "regularized"):
        torch.save({"pair_seed": 6099, "variant": variant, "joint_steps": 200_000,
                    "actor_model_version": trainer.STRUCTURED_MODEL_VERSION,
                    "feature_schema_sha256": trainer.STRUCTURED_FEATURE_SCHEMA_SHA256,
                    "bc_architecture_version": trainer.BC_ARCHITECTURE_VERSION,
                    "program_feedback_version": trainer.PROGRAM_FEEDBACK_VERSION,
                    "complete": True},
                   work / f"pair_6099_{variant}_matched.pt")
    metric = {"coverage": .27, "agreement": .95, "kl": .20,
              "illegal_count": 0, "rows": 120}
    role_hashes = {role: str(index) * 64 for index, role in enumerate(AGENTS, start=1)}
    program_hashes = {role: str(index) * 64 for index, role in enumerate(AGENTS, start=3)}
    audit = {role: {"audit_source": "preliminary_development_rollouts",
                    "development_seeds": list(range(240024, 240048)),
                    "sample_count": 1200,
                    "actor_role_sha256": role_hashes[role],
                    "program_sha256": program_hashes[role],
                    "max_features_per_node": 3,
                    "feedback_eligible": True,
                    "feedback": metric}
             for role in AGENTS}
    audit["robot_2"]["explanation_eligible"] = True
    audit["robot_2"]["explanation"] = metric
    acceptance = {
        "pair_seeds": [6099], "selected_pair_seed": 6099,
        "heldout_evaluated": False, "final_program_audit": {},
        "selected_regularized_feedback_rows": {role: 1 for role in AGENTS},
        "role_sha256": {"regularized": role_hashes},
        "program_sha256": {"regularized": program_hashes},
        "candidate_training": [{
            "pair_seed": 6099,
            "initial_role_sha256": role_hashes,
            "variants": {variant: {
                "joint_steps": 200_000, "actor_decisions": 400_000,
                "scenario_schedule_sha256": "a" * 64,
                "initial_role_sha256": role_hashes,
                "both_roles_changed": True, "post_policy_interventions": 0,
                "feedback_rows": ({role: 0 for role in AGENTS}
                                  if variant == "raw" else
                                  {role: 1 for role in AGENTS}),
            } for variant in ("raw", "regularized")},
        }],
        "development": [{"pair_seed": 6099,
                         "raw": {"passed": True},
                         "regularized": {"passed": True},
                         "preliminary_program_audit": audit}],
    }
    called = []
    def fake_run_experiment(**kwargs):
        called.append(kwargs)
        return acceptance
    monkeypatch.setattr(trainer, "run_experiment", fake_run_experiment)
    monkeypatch.setattr(sys, "argv", ["structured-runner", "--variant", "finalize-pilot",
                                       "--pair-seed", "6099", "--output-dir",
                                       str(tmp_path), "--development-pilot"])
    runner.main()
    result = json.loads(capsys.readouterr().out)
    assert called[0]["development_pilot"] is True
    assert called[0]["joint_steps"] == 200_000
    assert result["development_preflight"]["passed"] is True
    assert result["path"].endswith("warehouse_dual_structured_v54_acceptance.json")
