import gzip
import json
from pathlib import Path
from types import SimpleNamespace

from backend.training import warehouse_r4_active_trainer as active
from backend.training import warehouse_r4_role_feedback_update as role_feedback
from core.program import ExecutableProgram, ProgramNode
from env.warehouse.navigation import ACTIONS
from env.warehouse_native.feedback import FeedbackManager
from backend.training.warehouse_native_common import digest


ROOT = Path(__file__).parents[1]


def _real_transition():
    path = next((ROOT / "output/warehouse_native/alignment_50k_pair_20260910/branches/feedback/rollouts").glob("*.jsonl.gz"))
    with gzip.open(path, "rt") as stream:
        return json.loads(next(stream))


def test_r4_registered_mix_shaping_and_authority_contract():
    assert active.PARTNER_MIX == {
        "selfplay": .20, "skilled": .20, "assertive": .50, "noisy": .10,
    }
    assert active.SHAPING == {
        "productive_progress": .02,
        "avoidable_wait": -.03,
        "task_distance_regression": -.02,
    }
    assert active.KL_MAX == .01
    assert active.TRAINING_KL_LAMBDA == .001
    assert active.ACTOR_LR == active.CRITIC_LR == 1e-4
    assert active.ENTROPY_COEFFICIENT == .001
    assert active.ORDINARY_EPISODE_PROBABILITY == .85
    assert active.HARD_STATE_EPISODE_PROBABILITY == .15
    assert active.COLLISION_RECOVERY_EPISODE_PROBABILITY == 0.0
    assert active.LOW_ENERGY_EPISODE_PROBABILITY == 0.0
    assert active.LOW_ENERGY_ROBOT_2_BATTERY_MAX == 70.0
    assert active.TRAINING_PROGRAM_MIN_FIDELITY == .85
    assert active.TRAINING_PROGRAM_MIN_CRITICAL_FIDELITY == .80
    assert active.FINAL_EXPLANATION_MIN_FIDELITY == .90
    assert active.FINAL_EXPLANATION_MIN_CRITICAL_FIDELITY == .85
    assert sum(active.PARTNER_MIX.values()) == 1
    assert abs((active.ORDINARY_EPISODE_PROBABILITY
                + active.HARD_STATE_EPISODE_PROBABILITY
                + active.COLLISION_RECOVERY_EPISODE_PROBABILITY
                + active.LOW_ENERGY_EPISODE_PROBABILITY) - 1) < 1e-12


def test_r4_v18_hard_state_curriculum_is_train_only_unlabeled_and_immutable():
    scenarios = json.loads((ROOT /
        "output/warehouse_native/native_cycle_100k_20260909/scenarios.json").read_text())
    entries = scenarios["splits"]["train"][:8]
    before = json.dumps(entries, sort_keys=True)
    actor = ROOT / active.V8_STAGE_ROOT / "boundaries/step_0050000/actor.npz"
    rows, manifest = active._build_v8_public_hard_state_curriculum(
        entries, actor, maximum_scenarios=8, maximum_per_partner_group=8)
    assert rows
    assert json.dumps(entries, sort_keys=True) == before
    assert manifest["source_split"] == "train"
    assert manifest["validation_and_test_excluded"] is True
    assert manifest["source_actor_sha256"] == active.V8_STAGE_ACTOR_SHA256
    assert manifest["partners_called_before_actor"] is True
    assert manifest["source_snapshot_mutation"] is False
    assert manifest["action_labels_stored"] is False
    assert manifest["source_actor_commands_discarded"] is True
    assert manifest["source_action_overrides"] == 0
    assert manifest["ppo_joint_steps"] == 0
    assert set(manifest["admitted_partner_counts"]) == {
        "skilled", "assertive", "noisy",
    }
    assert set(manifest["admitted_category_counts"]) == {
        "mission_wait_or_regression", "charge_needed_nonprogress",
        "long_no_task_progress",
    }
    assert manifest["recoverability_check"]["public_state_only"] is True
    for row in rows:
        assert digest(row["snapshot"]) == row["source_snapshot_sha256"]
        assert row["recipe"]["source_split"] == "train"
        assert row["recipe"]["snapshot_mutation"] is False
        assert row["recipe"]["action_label_stored"] is False
        assert "action" not in row["recipe"]


def test_r4_v8_recovery_curriculum_preserves_exact_public_snapshots_and_authority():
    scenarios = json.loads((ROOT /
        "output/warehouse_native/native_cycle_100k_20260909/scenarios.json").read_text())
    entries = scenarios["splits"]["train"][:8]
    before = json.dumps(entries, sort_keys=True)
    actor = ROOT / active.V8_STAGE_ROOT / "boundaries/step_0050000/actor.npz"
    collision, energy, manifest = active._build_v8_program_partner_recovery_curriculum(
        entries, actor, maximum_scenarios=8)
    assert collision and energy
    assert json.dumps(entries, sort_keys=True) == before
    assert manifest["source_split"] == "train"
    assert manifest["validation_and_test_excluded"] is True
    assert manifest["source_actor_sha256"] == active.V8_STAGE_ACTOR_SHA256
    assert manifest["source_snapshot_mutation"] is False
    assert manifest["action_labels_stored"] is False
    assert manifest["partners_called_before_actor"] is True
    assert manifest["source_action_overrides"] == 0
    assert manifest["ppo_joint_steps"] == 0
    assert manifest["recoverability_check"]["public_state_only"] is True
    assert manifest["recoverability_check"]["action_labels_stored"] is False
    for row in (*collision, *energy):
        assert digest(row["snapshot"]) == row["source_snapshot_sha256"]
        assert row["recipe"]["action_label_stored"] is False
    assert all((row["snapshot"].get("public_feedback_history") or {}).get("collision_kind")
               for row in collision)
    assert all(active._goal_distance(row["snapshot"], 1)[1] for row in energy)


def test_r4_charge_margin_curriculum_is_public_reachable_and_does_not_mutate_source():
    scenarios = json.loads((ROOT /
        "output/warehouse_native/native_cycle_100k_20260909/scenarios.json").read_text())
    entry = scenarios["splits"]["train"][0]
    before = json.dumps(entry, sort_keys=True)
    row = active._derive_charge_margin_entry(entry)
    assert row is not None
    assert json.dumps(entry, sort_keys=True) == before
    source = entry["snapshot"]
    derived = row["snapshot"]
    restored = json.loads(json.dumps(derived))
    restored["state"]["agents"][1]["battery"] = source["state"]["agents"][1]["battery"]
    if "public_feedback_history" in source:
        restored["public_feedback_history"] = source["public_feedback_history"]
    assert restored == source
    assert row["recipe"]["changed_public_field"] == "state.agents[1].battery"
    assert row["recipe"]["initial_charge_needed"] is True
    assert row["recipe"]["reachable_without_shutdown"] is True
    assert derived["state"]["agents"][1]["battery"] >= (
        row["recipe"]["charger_distance"] * row["recipe"]["move_battery_cost"])
    assert active._goal_distance(derived, 1)[1] is True


def test_r4_trajectory_curriculum_uses_train_only_and_exact_r3_commands():
    scenarios = json.loads((ROOT /
        "output/warehouse_native/native_cycle_100k_20260909/scenarios.json").read_text())
    entries = scenarios["splits"]["train"][:8]
    before = json.dumps(entries, sort_keys=True)
    actor = ROOT / active.SOURCE_ROOT / "branches/feedback/actors/actor_0050000.npz"
    rows, manifest = active._build_trajectory_charge_margin_curriculum(
        entries, actor, maximum_scenarios=8, maximum_frames=16, maximum_rows=64)
    assert rows
    assert json.dumps(entries, sort_keys=True) == before
    assert manifest["source_split"] == "train"
    assert manifest["validation_and_test_excluded"] is True
    assert manifest["source_actor_sha256"] == active.SOURCE_ACTOR_SHA256
    assert manifest["ppo_joint_steps"] == 0
    assert manifest["source_neural_actions"] == 2 * manifest["evidence_environment_steps"]
    assert manifest["source_action_overrides"] == 0
    assert manifest["robot_2_position_coverage_count"] >= 1
    assert manifest["charger_distance_counts"]


def test_r4_shaping_reads_real_public_transition_without_mutation():
    transition = _real_transition()
    before = json.dumps(transition, sort_keys=True)
    for role in range(2):
        value, flags = active.shaping_for_transition(transition, role)
        assert value in (-.05, -.03, -.02, 0., .02)
        assert set(flags) == {
            "productive", "safe_task_progress", "avoidable_wait",
            "distance_regression", "charge_needed",
        }
    assert json.dumps(transition, sort_keys=True) == before


def test_r4_source_artifacts_are_exact():
    root = ROOT / active.SOURCE_ROOT
    marker = json.loads((root / "branches/feedback/boundaries/step_0050000.json").read_text())
    assert marker["actor"]["sha256"] == active.SOURCE_ACTOR_SHA256
    assert marker["checkpoint"]["sha256"] == active.SOURCE_CHECKPOINT_SHA256
    assert active.file_hash(root / marker["actor"]["path"]) == active.SOURCE_ACTOR_SHA256
    assert active.file_hash(root / marker["checkpoint"]["path"]) == active.SOURCE_CHECKPOINT_SHA256


def test_r4_feedback_kl_is_scoped_to_deployed_robot_2():
    from tests.test_warehouse_family_stable_ppo_update import Native, batch, program
    trainer = Native(epochs=1, minibatch=8)
    result = role_feedback.stable_ppo_update(
        trainer, batch(), program(), lambda_value=.01,
        auxiliary_observations=None, feedback_role=1,
    )
    audit = result["stable_update"]
    assert audit["feedback_role"] == 1
    assert audit["ppo_nn_row_visits"] == 6
    assert audit["ordinary_tree_rows"] == audit["ordinary_kl_row_visits"] == 3
    assert result["feedback_loss"] > 0


def test_r4_relational_feedback_transform_is_public_observed197_only():
    import numpy as np
    from backend.training import warehouse_family_relational_tree_features as relations

    names = relations.ORIGINAL_NAMES
    original = np.zeros((3, 197), dtype=np.float32)
    original[:, names.index("self.battery")] = (1.0, .6, .2)
    original[:, names.index("charger.self.path_distance")] = (.1, .2, .3)
    transformed = relations.transform(original, names)
    assert transformed.shape == (3, 268)
    margin = relations.names(names).index(
        "relation.charger.self.battery_minus_encoded_move_cost")
    assert np.allclose(transformed[:, margin], (0.918, .436, -.046))
    assert np.array_equal(original[:, :], transformed[:, :197])


def test_r4_initial_feedback_requires_exact_r3_actor_and_observed197():
    from backend.training import warehouse_family_relational_tree_features as relations
    names = relations.ORIGINAL_NAMES
    manager = FeedbackManager(names, active.TREE_CONFIG)
    manager.program = ExecutableProgram(
        tuple(ACTIONS), names,
        ProgramNode(probabilities=(.2, .2, .2, .2, .2)),
        {"native_source_actor_sha256": active.SOURCE_ACTOR_SHA256},
    )
    manager.reliable = True
    artifact = {
        "version": "warehouse-r4-r3-preextraction.v1",
        "test_fixture": False,
        "source_actor_sha256": active.SOURCE_ACTOR_SHA256,
        "extraction_environment_steps": 16,
        "ppo_joint_steps": 0,
        "program_content_sha256": digest(manager.program.to_dict()),
        "feedback_manager": manager.state_dict(),
    }
    trainer = object.__new__(active.ActiveTrainer)
    trainer.joint_steps = 0
    trainer.current_lambda = 0.
    trainer.feedback_manager = FeedbackManager(names, active.TREE_CONFIG)
    trainer.envs = [SimpleNamespace(feature_names=names)]
    trainer.protocol = {"initial_feedback": None}
    trainer.bind_initial_feedback(artifact, artifact_sha256="a" * 64, lambda_value=.005)
    assert trainer.current_lambda == .005
    assert trainer.protocol["initial_feedback"]["source_actor_sha256"] == active.SOURCE_ACTOR_SHA256
    assert trainer.feedback_manager.program.metadata["native_source_actor_sha256"] == active.SOURCE_ACTOR_SHA256
