"""Synthetic capability tests; real scene collection and BC are never run."""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from backend.training import warehouse_v621_bc_capability as c
from experiments.warehouse import v621_bc_protocol as p
from experiments.warehouse import v621_protocol as d


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return {"path": str(path), "sha256": c.file_sha(path)}


@pytest.fixture
def spec_fixture(tmp_path, monkeypatch):
    root = tmp_path
    for relative in c.MODEL_CODE_FILES:
        target = root/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(relative)
    sources = {name: c.file_sha(root/name) for name in c.MODEL_CODE_FILES}
    monkeypatch.setattr(p, "source_files", lambda _: dict(sources))
    monkeypatch.setattr(p, "checked_configuration", lambda _: None)
    monkeypatch.setattr(c, "_checked_prerequisites", lambda *_: {})
    monkeypatch.setattr(c, "_checked_shared_cache", lambda *_: {})
    # Tiny schedule tests exercise order/consumption, while a separate test
    # verifies the unpatched real declared schedule's exact 542/1005 counts.
    schedule = [{"phase": "stage", "scenario_seed": 42, "action_seed": 99},
                {"phase": "stage", "scenario_seed": 43, "action_seed": 100}]
    monkeypatch.setattr(c, "reset_schedule", lambda *_: list(schedule))
    recipe = write(root/"configs/recipe.json", p.recipe())
    seeds = write(root/"configs/seeds.json", p.seed_plan())
    source_sha = c.digest(sources)
    model_sha = c.model_implementation_sha256(root)
    clearance = {"schema": c.CLEARANCE_SCHEMA, "candidate_id": d.CANDIDATE_ID,
        "batch_id": p.BATCH, "approved_for_bc_development": True,
        "source_closure_sha256": source_sha, "model_implementation_sha256": model_sha,
        "baseline_actor_sha256": d.ACTOR_SHA256, "recipe_sha256": recipe["sha256"],
        "seed_plan_sha256": seeds["sha256"], "authorized_reset_contract_sha256": c.digest(c.reset_contract()),
        "initializations": list(p.INITIALIZATIONS), "unresolved": [], "remaining_conflicts": [],
        "fresh_precheck_authorized": False, "PPO_joint_steps": 0, "pilot_opened": False, "publishable": False}
    spec = {"schema": c.SCHEMA, "candidate_id": d.CANDIDATE_ID, "batch_id": p.BATCH,
        "source_closure_sha256": source_sha, "model_implementation_sha256": model_sha,
        "recipe": recipe, "seed_plan": seeds, "seed_clearance": write(root/"evidence/clearance.json", clearance),
        "output_directory": str(root/"output/warehouse_v621"/p.BATCH)}
    spec_path = root/"spec.json"
    write(spec_path, spec)
    return root, spec_path, spec, schedule


def test_real_declared_schedule_keeps_shared_teacher_critic_and_model_visits_separate():
    for initialization in p.INITIALIZATIONS:
        cache = c.reset_schedule(initialization, "shared_cache")
        assert len(cache) == 542
        assert [r["phase"] for r in cache[-2:]] == ["critic_setup", "critic_setup"]
        assert cache[-1] == cache[-2]
        assert cache[-1]["action_seed"] is None
        assert cache[-1]["scenario_seed"] == p.seed_plan()["initializations"][str(initialization)]["critic_seed"]
        assert {r["phase"] for r in cache[:-2]} == set(p.TEACHER_STAGES)
        control = c.reset_schedule(initialization, "additive_control")
        main = c.reset_schedule(initialization, "shared_b035")
        backup = c.reset_schedule(initialization, "shared_b050")
        assert control == main == backup and len(main) == 1005
        assert all(r["phase"] not in p.TEACHER_STAGES for r in main[:730])
        assert len({(r["scenario_seed"], r["action_seed"]) for r in main[:730]}) == 730
        assert main[730:] == [{"phase": "spent_development_evaluation",
            "scenario_seed": r["scenario_seed"], "action_seed": r["action_seed"]} for r in d.source_cases()]


def test_ordinary_construction_training_and_fake_plan_are_rejected():
    with pytest.raises(RuntimeError, match="ordinary"):
        c.require_active_plan(operation="bc_update")
    with pytest.raises(RuntimeError, match="operation"):
        c.require_active_plan(operation="PPO")
    with pytest.raises(RuntimeError):
        c.BCDevelopmentPlan(Path('/tmp'), Path('/tmp/spec'), 6171, 'shared_b035', 'model',
            Path('/tmp/out'), Path('/tmp/journal'), (), 'a', 'b', 'c', {}, {}, object())


def test_authorized_exact_order_is_journaled_before_caller_can_reset_and_single_use(spec_fixture):
    root, path, _, schedule = spec_fixture
    with c.load_and_authorize(root, path, "shared_b035", 6171) as plan:
        assert c.require_active_plan(plan, operation="bc_update") is plan
        assert plan.init == plan.initialization == 6171
        for index, expected in enumerate(schedule):
            ticket = plan.authorize_reset(expected["phase"], expected["scenario_seed"], expected["action_seed"])
            rows = [json.loads(line) for line in plan.journal_path.read_text().splitlines()]
            assert len(rows) == index+1 and rows[-1] == ticket
            assert ticket["consumed_before_actual_reset"]
            assert ticket["not_an_actual_reset_receipt"]
        plan.assert_completed()
    assert (plan.output_directory/"capability.completed.json").exists()
    with pytest.raises(FileExistsError):
        with c.load_and_authorize(root, path, "shared_b035", 6171):
            pass
    with pytest.raises(RuntimeError, match="ordinary"):
        c.require_active_plan(plan, operation="export")


@pytest.mark.parametrize("kind", ["phase", "scene", "action", "extra", "incomplete", "source", "evidence"])
def test_out_of_order_incomplete_or_changed_inputs_close_attempt_without_retry(spec_fixture, kind):
    root, path, spec, schedule = spec_fixture
    with pytest.raises(RuntimeError):
        with c.load_and_authorize(root, path, "shared_b035", 6171) as plan:
            row = schedule[0]
            if kind == "phase":
                plan.authorize_reset("wrong", row["scenario_seed"], row["action_seed"])
            elif kind == "scene":
                plan.authorize_reset(row["phase"], 43, row["action_seed"])
            elif kind == "action":
                plan.authorize_reset(row["phase"], row["scenario_seed"], 100)
            elif kind == "source":
                (root/c.MODEL_CODE_FILES[0]).write_text("tampered")
                plan.assert_immutable()
            elif kind == "evidence":
                Path(spec["seed_clearance"]["path"]).write_text("tampered")
                plan.assert_immutable()
            elif kind == "extra":
                for expected in schedule:
                    plan.authorize_reset(expected["phase"], expected["scenario_seed"], expected["action_seed"])
                plan.authorize_reset("stage", 44, 101)
            else:
                plan.authorize_reset(row["phase"], row["scenario_seed"], row["action_seed"])
    assert (plan.output_directory/"capability.failed.json").exists()
    assert not (plan.output_directory/"capability.completed.json").exists()


@pytest.mark.parametrize("key,value", [("approved_for_bc_development", False), ("approved_for_bc_development", 1),
    ("unresolved", ["unknown opaque producer"]), ("remaining_conflicts", ["earlier reset"]),
    ("source_closure_sha256", "wrong"), ("model_implementation_sha256", "wrong"),
    ("authorized_reset_contract_sha256", "wrong"), ("PPO_joint_steps", False),
    ("fresh_precheck_authorized", True), ("pilot_opened", True)])
def test_seed_clearance_requires_exact_typed_closed_source_model_and_schedule_bindings(spec_fixture, key, value):
    root, path, spec, _ = spec_fixture
    ledger_path = Path(spec["seed_clearance"]["path"])
    ledger = json.loads(ledger_path.read_text())
    ledger[key] = value
    spec["seed_clearance"] = write(ledger_path, ledger)
    write(path, spec)
    with pytest.raises(RuntimeError, match="seed clearance"):
        with c.load_and_authorize(root, path, "shared_b035", 6171):
            pass
    assert not (root/"output/warehouse_v621"/p.BATCH).exists()


def test_shared_cache_does_not_allow_policy_updates_and_contexts_do_not_nest(spec_fixture):
    root, path, _, schedule = spec_fixture
    with c.load_and_authorize(root, path, "shared_cache", 6171) as plan:
        assert c.require_active_plan(plan, operation="critic_update") is plan
        with pytest.raises(RuntimeError, match="shared cache"):
            c.require_active_plan(plan, operation="bc_update")
        with pytest.raises(RuntimeError, match="nest"):
            with c.load_and_authorize(root, path, "shared_b035", 6171):
                pass
        for row in schedule:
            plan.authorize_reset(row["phase"], row["scenario_seed"], row["action_seed"])


def test_backup_requires_all_three_main_results_and_real_quality_failure(spec_fixture):
    root, path, spec, schedule = spec_fixture
    with pytest.raises(RuntimeError, match="three completed"):
        with c.load_and_authorize(root, path, "shared_b050", 6171):
            pass
    refs = {}
    for initialization in p.INITIALIZATIONS:
        folder = root/"output/warehouse_v621"/p.BATCH/f"init-{initialization}"/"shared_b035"
        folder.mkdir(parents=True)
        native = folder/"native-successful-resets.jsonl"
        native.write_text(''.join(json.dumps({"phase": "bc", "seed": r["scenario_seed"]})+'\n' for r in schedule))
        pre = folder/"reset-preauthorization.jsonl"
        pre.write_text(''.join(json.dumps(dict(r, index=i))+'\n' for i, r in enumerate(schedule)))
        capability = write(folder/"capability.completed.json", {"preauthorized_resets": len(schedule),
            "initialization": initialization, "recipe": "shared_b035", "preauthorization_journal_sha256": c.file_sha(pre)})
        collection = write(folder/"development.collection.json", {"cases": d.source_cases(),
            "outcomes": [dict(case=case, deliveries=6, robot_collisions=0, shutdown=False, physical_frames=120)
                         for case in d.source_cases()]})
        main = {"schema": "warehouse-v621-bc-development-result.v1", "status": "completed",
            "recipe": "shared_b035", "initialization": initialization, "quality_passed": True,
            "program_passed": False, "source_closure_sha256": spec["source_closure_sha256"],
            "seed_plan_sha256": spec["seed_plan"]["sha256"], "model_implementation_sha256": spec["model_implementation_sha256"],
            "fresh_precheck_episodes": 0, "PPO_joint_steps": 0, "publishable": False,
            "actual_reset_count": len(schedule), "success_journal": {"path": str(native), "sha256": c.file_sha(native)},
            "preauthorization_journal": {"path": str(pre), "sha256": c.file_sha(pre)},
            "development_collection": collection,
            "quality": {"mean_deliveries": 6., "mean_collisions": 0., "shutdown_episode_rate": 0.}}
        refs[str(initialization)] = write(folder/"result.json", main)
        write(folder/"completed.json", {"result": refs[str(initialization)], "capability_completed": capability,
            "actual_reset_count": len(schedule)})
    spec["main_results"] = refs
    write(path, spec)
    with pytest.raises(RuntimeError, match="program-only"):
        with c.load_and_authorize(root, path, "shared_b050", 6171):
            pass
    main_path = Path(refs["6172"]["path"])
    main = json.loads(main_path.read_text())
    main["quality_passed"] = False
    # A flag alone is rejected, even with a newly bound result SHA.
    refs["6172"] = write(main_path, main)
    completed_path = main_path.parent/"completed.json"
    completed = json.loads(completed_path.read_text())
    completed["result"] = refs["6172"]
    write(completed_path, completed)
    write(path, spec)
    with pytest.raises(RuntimeError, match="does not recompute"):
        with c.load_and_authorize(root, path, "shared_b050", 6171):
            pass
    collection_path = Path(main["development_collection"]["path"])
    collection = json.loads(collection_path.read_text())
    for row in collection["outcomes"]:
        if row["case"]["original_split"] == "validation":
            row["deliveries"] = 5
    main["development_collection"] = write(collection_path, collection)
    main["quality"]["mean_deliveries"] = 5.
    refs["6172"] = write(main_path, main)
    completed["result"] = refs["6172"]
    write(completed_path, completed)
    write(path, spec)
    with c.load_and_authorize(root, path, "shared_b050", 6171) as plan:
        for row in schedule:
            plan.authorize_reset(row["phase"], row["scenario_seed"], row["action_seed"])


def test_caught_invalid_reset_cannot_be_repaired_inside_the_same_capability(spec_fixture):
    root, path, _, _ = spec_fixture
    with pytest.raises(RuntimeError, match="invalidated"):
        with c.load_and_authorize(root, path, "shared_b035", 6171) as plan:
            with pytest.raises(RuntimeError, match="ordered"):
                plan.authorize_reset("wrong", 42, 99)
            with pytest.raises(RuntimeError, match="ordinary"):
                c.require_active_plan(plan, operation="bc_update")
    assert (plan.output_directory/"capability.failed.json").exists()


def test_cache_verification_checks_all_assets_semantic_data_hashes_and_native_reset_order(tmp_path):
    import numpy as np
    root, initialization = tmp_path, 6171
    directory = root/"output/warehouse_v621"/p.BATCH/f"init-{initialization}"/"shared_cache"
    directory.mkdir(parents=True)
    spec = {"source_closure_sha256": "a"*64, "seed_plan": {"sha256": "b"*64},
            "model_implementation_sha256": "c"*64}
    assets = {}
    for stage in p.TEACHER_STAGES:
        path = directory/f"{stage}.npz"
        np.savez_compressed(path, **{"0": np.asarray([[1., 2.]], np.float32)})
        assets[stage] = {"path": str(path), "sha256": c.file_sha(path), "data_sha256": c._array_file_identity(path)}
        path.chmod(0o444)
    critic = directory/"collision-critic.pt"
    critic.write_bytes(b"opaque torch state; file hashing must not pretend this is NPZ")
    critic.chmod(0o444)
    critic_asset = {"path": str(critic), "sha256": c.file_sha(critic), "model_sha256": "d"*64}
    critic_report = write(directory/"critic-report.json", {"model_sha256": "d"*64})
    schedule = c.reset_schedule(initialization, "shared_cache")
    native = directory/"success.jsonl"
    native.write_text(''.join(json.dumps({"phase": "bc", "seed": r["scenario_seed"]})+'\n' for r in schedule))
    pre = directory/"reset-preauthorization.jsonl"
    pre.write_text(''.join(json.dumps(dict(row, index=i))+'\n' for i, row in enumerate(schedule)))
    write(directory/"capability.started.json", {"schema": "warehouse-v621-bc-development-capability.v1",
        "source_closure_sha256": spec["source_closure_sha256"], "initialization": initialization,
        "recipe": "shared_cache", "scheduled_reset_count": 542})
    write(directory/"capability.completed.json", {"schema": "warehouse-v621-bc-capability-completed.v1",
        "initialization": initialization, "recipe": "shared_cache", "preauthorized_resets": 542,
        "preauthorization_journal_sha256": c.file_sha(pre)})
    write(directory/"cache.manifest.json", {"schema": "warehouse-v621-bc-shared-cache.v1",
        "status": "completed", "initialization": initialization,
        "source_closure_sha256": spec["source_closure_sha256"], "seed_plan_sha256": spec["seed_plan"]["sha256"],
        "model_implementation_sha256": spec["model_implementation_sha256"], "actual_reset_count": 542,
        "actor_updates": 0, "PPO_joint_steps": 0, "teacher_assets": assets, "critic_asset": critic_asset,
        "critic_report": critic_report, "success_journal": {"path": str(native), "sha256": c.file_sha(native)}})
    write(directory/"completed.json", {"result": {"path": str(directory/"cache.manifest.json"), "sha256": c.file_sha(directory/"cache.manifest.json")},
        "capability_completed": {"path": str(directory/"capability.completed.json"), "sha256": c.file_sha(directory/"capability.completed.json")},
        "source_closure_sha256": spec["source_closure_sha256"], "actual_reset_count": 542})
    result = c._checked_shared_cache(root, spec, initialization)
    assert len(result) == 12
    native.write_text(native.read_text().replace(str(schedule[0]["scenario_seed"]), "999", 1))
    with pytest.raises(RuntimeError, match="journal changed"):
        c._checked_shared_cache(root, spec, initialization)
