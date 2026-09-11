import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backend.training import warehouse_r41_active_evaluation as evaluation
from backend.training import warehouse_r41_active_run as run
from backend.training import warehouse_r41_active_trainer as trainer
from backend.training import warehouse_r41_training_ledger as ledger
from backend.warehouse_r41_online_runtime import R41ConflictWarehouseEnv
from env.warehouse_native.policy import NumPyNativeActor
from env.warehouse_native.r41_conflict import (
    CONFLICT_GRAPH_SHA256, CONTRACT_SHA256, reset_r41_scenario,
    r41_scene_fingerprint,
)


ROOT = Path(__file__).parents[1]


def conflict_api():
    return SimpleNamespace(
        R41ConflictWarehouseEnv=R41ConflictWarehouseEnv,
        reset_conflict_scenario=reset_r41_scenario,
        validate_conflict_manifest=lambda value: value,
        CONTRACT_VERSION="fixture",
        CONTRACT_SHA256=CONTRACT_SHA256,
        CONFLICT_GRAPH_SHA256=CONFLICT_GRAPH_SHA256,
        MANIFEST_VERSION="fixture",
    )


def scene(seed, index=0):
    env = R41ConflictWarehouseEnv()
    env.reset(seed=seed)
    return {
        "id": f"fixture_{index}", "split": "train", "seed": seed,
        "fingerprint": r41_scene_fingerprint(env), "snapshot": env.snapshot(),
        "contract_sha256": CONTRACT_SHA256,
        "conflict_graph_sha256": CONFLICT_GRAPH_SHA256,
    }


def test_r41_training_contract_is_fresh_r3_bounded_and_neural_only():
    assert trainer.SOURCE_ACTOR_SHA256 == (
        "309b6e53fe682bead8d3443015aca27eae60e561175e71d7c25f57314ac69d5b")
    assert trainer.MAXIMUM_ADDITIONAL_JOINT_STEPS == 2_000_000
    assert trainer.BOUNDARY_INTERVAL == 50_000
    assert trainer.PARTNER_MIX == {
        "selfplay": .20, "skilled": .20, "assertive": .50, "noisy": .10,
    }
    assert trainer.SHAPING == {
        "productive_progress": .02,
        "avoidable_wait": -.03,
        "task_distance_regression": -.02,
    }
    assert trainer.ORDINARY_EPISODE_PROBABILITY == .80
    assert trainer.REACHABLE_ENERGY_EPISODE_PROBABILITY == .20
    instance = object.__new__(trainer.R41ActiveTrainer)
    with pytest.raises(ValueError, match="fresh r3 child"):
        instance.bind_r4_stage_source({})


def test_reachable_energy_state_requires_public_margin():
    env = R41ConflictWarehouseEnv()
    env.reset(seed=841_001)
    actor = env.state.agents[1]
    accepted = None
    for battery in range(2, 71, 2):
        actor.battery = float(battery)
        result = trainer._reachable_energy_state(env.snapshot())
        if result is not None:
            accepted = result
            break
    assert accepted is not None
    assert accepted["charge_needed"] is True
    assert accepted["reachable_without_shutdown"] is True
    assert accepted["public_safety_margin"] >= accepted["configured_safety_margin"]
    actor.battery = max(0., accepted["required_battery_to_charger"] - 2.)
    assert trainer._reachable_energy_state(env.snapshot()) is None


def test_energy_curriculum_is_exact_train_only_unlabelled_and_zero_override(monkeypatch):
    if not trainer.SOURCE_ACTOR_PATH.is_file():
        pytest.skip("requires the private frozen r3 source Actor")
    monkeypatch.setattr(trainer, "_conflict_api", conflict_api)
    monkeypatch.setattr(trainer, "ENERGY_MINIMUM_ROWS_PER_PARTNER", 1)
    monkeypatch.setattr(trainer, "partner_action", lambda *args, **kwargs: "WAIT")

    class BackAndForthActor:
        def __init__(self, path):
            self.calls = 0

        def act(self, observations, deterministic=True):
            action = "UP" if self.calls % 2 == 0 else "DOWN"
            self.calls += 1
            index = trainer.ACTIONS.index(action)
            probabilities = {}
            actions = {}
            for agent_id in observations:
                values = np.zeros(len(trainer.ACTIONS), dtype=np.float32)
                values[index] = 1.
                probabilities[agent_id] = values
                actions[agent_id] = action
            return actions, probabilities

    monkeypatch.setattr(trainer, "NumPyNativeActor", BackAndForthActor)
    entries = [scene(842_000 + index, index) for index in range(9)]
    before = json.dumps(entries, sort_keys=True)
    rows, report = trainer.build_reachable_energy_curriculum(
        entries, maximum_source_episodes=27, rows_per_partner=2)
    assert len(rows) == 6
    assert report["rows_by_partner"] == {"skilled": 2, "assertive": 2, "noisy": 2}
    assert report["source_split"] == "train"
    assert report["snapshot_mutation"] is False
    assert report["action_labels_stored"] is False
    assert report["action_overrides"] == 0
    assert report["actor_commands"] == report["actor_action_equal"]
    assert json.dumps(entries, sort_keys=True) == before
    for row in rows:
        assert row["action_label_stored"] is False
        assert "action" not in row
        assert trainer.digest(row["snapshot"]) == row["source_snapshot_sha256"]
        assert row["admission"]["public_safety_margin"] >= 4.


def summary(**updates):
    result = {
        "active_non_wait_rate": .90,
        "productive_action_rate": .75,
        "full_battery_non_charger_wait_rate": .01,
        "first_productive_latency_median": 1.,
        "first_productive_latency_p90": 3.,
        "ai_delivery_share": .50,
        "no_task_progress_streak_p95": 12.,
        "collision_cancellation_rate": .05,
        "mean_longest_collision_streak": 2.,
        "static_wall_command_rate": 0.,
        "ai_shutdown_count": 0,
        "action_override_count": 0,
        "policy_action_equality_rate": 1.,
        "mean_ai_deliveries": 4.,
    }
    result.update(updates)
    return result


def test_dual_suite_gate_requires_four_relative_gains_and_all_absolute():
    baseline = summary(active_non_wait_rate=.80, productive_action_rate=.60,
                       mean_ai_deliveries=2., no_task_progress_streak_p95=20.)
    passed = evaluation._suite_decision(baseline, summary())
    assert passed["selected"] is True
    assert passed["relative_checks_passed"] == 5
    failed = evaluation._suite_decision(
        baseline, summary(ai_shutdown_count=1))
    assert failed["selected"] is False
    assert failed["absolute_checks"]["shutdowns"] is False


def test_conflict_episode_submits_exact_r3_actor_action(monkeypatch):
    if not trainer.SOURCE_ACTOR_PATH.is_file():
        pytest.skip("requires the private frozen r3 source Actor")
    monkeypatch.setattr(evaluation, "_conflict_api", conflict_api)
    entry = scene(843_001)
    actor = NumPyNativeActor(trainer.SOURCE_ACTOR_PATH)
    row = evaluation.evaluate_conflict_episode(actor, entry, "fixed_yield", seed=9)
    assert row["policy_actions"] == row["submitted_actions"] == row["action_equal"]
    assert row["action_overrides"] == 0
    assert row["conflict_contract_sha256"] == CONTRACT_SHA256


def test_deterministic_actor_validation_rejects_non_argmax():
    class BadActor:
        def act(self, observations, deterministic=True):
            probabilities = {key: np.asarray([.1, .1, .1, .6, .1])
                             for key in observations}
            return {key: "WAIT" for key in observations}, probabilities
    with pytest.raises(RuntimeError, match="deterministic output"):
        evaluation._actor_actions(BadActor(), {"robot_1": [0], "robot_2": [0]})


def test_target_validation_never_allows_more_than_two_million(tmp_path):
    # Argument validation happens before any source loader or filesystem write.
    with pytest.raises(ValueError, match="no larger than 2M"):
        run.execute(tmp_path / "run", target=2_050_000,
                    conflict_manifest_path=tmp_path / "missing",
                    original_scenarios_path=tmp_path / "missing")


def test_saved_ledger_fails_closed_without_selected_actor(tmp_path):
    path = tmp_path / "ledger.json"
    value = {
        "version": ledger.VERSION, "status": "failed_no_eligible_actor",
        "admission_eligible": False,
        "source": {"actor_sha256": trainer.SOURCE_ACTOR_SHA256,
                   "failed_r4_parent_used": False},
        "maximum_additional_joint_steps": 2_000_000,
        "total_actual_additional_joint_steps": 2_000_000,
        "runtime_action_override": False, "selected": None,
    }
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="no eligible Actor"):
        ledger.read_saved_ledger(path, expected_sha256=ledger.file_hash(path),
                                 require_selected=True)
    assert ledger.read_saved_ledger(
        path, expected_sha256=ledger.file_hash(path),
        require_selected=False)["admission_eligible"] is False


def test_historical_r3_restore_allows_only_registered_additive_source(
        tmp_path, monkeypatch):
    from backend.training import warehouse_family_alignment_run as alignment_run
    from backend.training import warehouse_native_partner_mix_evaluation as compact

    additive = "env/warehouse_native/r41_conflict.py"
    old = {"historical.py": "a" * 64}
    live = {**old, additive: trainer.file_hash(ROOT / additive)}
    original = lambda: dict(live)
    monkeypatch.setattr(compact, "execution_sources", original)
    monkeypatch.setattr(alignment_run, "sources", lambda: compact.execution_sources())
    (tmp_path / "prepared.json").write_text(json.dumps({
        "identity": {"runtime_sources": old},
    }))
    with trainer._r3_historical_source_compatibility(tmp_path) as receipt:
        assert alignment_run.sources() == old
        assert receipt["historical_runtime_sources_sha256"] == trainer.digest(old)
        assert receipt["additive_sources"] == {additive: live[additive]}
    assert compact.execution_sources() == live

    monkeypatch.setattr(
        compact, "execution_sources",
        lambda: {**live, "env/warehouse_native/unregistered.py": "b" * 64},
    )
    with pytest.raises(ValueError, match="exact r3 closure"):
        with trainer._r3_historical_source_compatibility(tmp_path):
            pass


def test_run_source_receipt_binds_manifest_contract_graph_and_version(tmp_path):
    conflict_path = tmp_path / "conflict.json"
    conflict = {"version": "fixture-manifest", "splits": {}}
    conflict_path.write_text(json.dumps(conflict))
    original_path = tmp_path / "original.json"
    original_path.write_text(json.dumps({
        "splits": {"validation": [{"id": index} for index in range(50)]},
    }))
    fake = SimpleNamespace(
        source_state_sha256="c" * 64,
        protocol={"scenario_sampling": {
            "manifest_version": "fixture-manifest",
            "conflict_contract_sha256": CONTRACT_SHA256,
            "conflict_contract_version": "fixture-contract",
            "conflict_graph_sha256": CONFLICT_GRAPH_SHA256,
        }},
    )
    receipt = run._source_receipt(fake, conflict_path, conflict, original_path)
    binding = receipt["conflict_manifest"]
    assert binding["version"] == "fixture-manifest"
    assert binding["contract_sha256"] == CONTRACT_SHA256
    assert binding["contract_version"] == "fixture-contract"
    assert binding["graph_sha256"] == CONFLICT_GRAPH_SHA256
    assert binding["semantic_sha256"] == trainer.digest(conflict)
    assert receipt["failed_r4_parent_used"] is False
    assert receipt["runtime_action_override"] is False


def test_ledger_accepts_complete_zero_boundary_receipt_and_rejects_missing_graph(
        tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    original = {
        "file_sha256": "d" * 64,
        "validation_entries_sha256": "e" * 64,
        "count": 50,
    }
    sampling = {
        "manifest_version": "fixture-manifest",
        "manifest_file_sha256": "f" * 64,
        "manifest_semantic_sha256": "1" * 64,
        "conflict_contract_sha256": CONTRACT_SHA256,
        "conflict_contract_version": "fixture-contract",
        "conflict_graph_sha256": CONFLICT_GRAPH_SHA256,
        "train_entries_sha256": "2" * 64,
    }
    protocol = {
        "version": trainer.VERSION,
        "source": {
            "actor_sha256": trainer.SOURCE_ACTOR_SHA256,
            "checkpoint_sha256": trainer.SOURCE_CHECKPOINT_SHA256,
            "cumulative_joint_steps": trainer.SOURCE_CUMULATIVE_STEPS,
            "failed_r4_parent_used": False,
        },
        "partners": trainer.PARTNER_MIX,
        "shaping": trainer.SHAPING,
        "implementation_sources": trainer._implementation_sources(),
        "maximum_additional_joint_steps": trainer.MAXIMUM_ADDITIONAL_JOINT_STEPS,
        "runtime_action_override": False,
        "scenario_sampling": sampling,
        "evaluation": {
            "original_validation": original,
            "conflict_validation_entries_sha256": "3" * 64,
        },
    }
    receipt = {
        "version": run.RUN_VERSION,
        "source_actor_sha256": trainer.SOURCE_ACTOR_SHA256,
        "source_checkpoint_sha256": trainer.SOURCE_CHECKPOINT_SHA256,
        "source_state_sha256": "4" * 64,
        "source_cumulative_joint_steps": trainer.SOURCE_CUMULATIVE_STEPS,
        "failed_r4_parent_used": False,
        "runtime_action_override": False,
        "original_validation": original,
        "conflict_manifest": {
            "file_sha256": sampling["manifest_file_sha256"],
            "semantic_sha256": sampling["manifest_semantic_sha256"],
            "version": sampling["manifest_version"],
            "contract_sha256": sampling["conflict_contract_sha256"],
            "contract_version": sampling["conflict_contract_version"],
            "graph_sha256": sampling["conflict_graph_sha256"],
        },
    }
    terminal = {
        "version": run.RUN_VERSION,
        "status": "target_completed_no_selection",
        "current_additional_joint_steps": 0,
        "cumulative_joint_steps": trainer.SOURCE_CUMULATIVE_STEPS,
        "maximum_additional_joint_steps": trainer.MAXIMUM_ADDITIONAL_JOINT_STEPS,
        "runtime_action_override": False,
        "selected": False,
    }
    for name, value in (("protocol.json", protocol),
                        ("source_receipt.json", receipt),
                        ("run.json", terminal)):
        (root / name).write_text(json.dumps(value))
    report = ledger.build(root, tmp_path / "ledger.json")
    assert report["status"] == "failed_no_eligible_actor"
    assert report["conflict_graph_sha256"] == CONFLICT_GRAPH_SHA256

    receipt["conflict_manifest"].pop("graph_sha256")
    (root / "source_receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="conflict contract"):
        ledger.build(root, tmp_path / "second-ledger.json")
