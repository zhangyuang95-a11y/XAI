"""No-reset protocol checks for the v6.14 exploratory Warehouse candidate."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.warehouse import protocol as p  # noqa: E402


def _receipt(pair_seed: int = 12469) -> dict:
    schedule = p.bc_data_seed_schedule(pair_seed)
    critic_reset = p.bc_critic_reset_schedule(pair_seed, p.INITIAL_RELEASE_ID)
    actor_roles = {role: sha256(role.encode()).hexdigest() for role in p.ROLES}
    warmup_roles = {role: sha256((role + ":warmup").encode()).hexdigest()
                    for role in p.ROLES}
    warmup_schedule = {name.removeprefix("topology_"): schedule[name]
                       for name in schedule if name.startswith("topology_")}
    final_schedule = {name.removeprefix("final_bc_precheck_"): schedule[name]
                      for name in schedule if name.startswith("final_bc_precheck_")}
    stages = []
    for name, epochs in p.bc_stage_receipt_shape():
        row = {"stage": name}
        if epochs is None:
            row.update({"episode_seeds": schedule[name],
                        "kind": ("public_teacher" if "teacher" in name
                                 else "dual_nn_visitation"),
                        "post_policy_interventions": 0})
            if "argmax" in name or "sampled" in name:
                row["training_only_argmax_visits"] = "argmax" in name
        elif name.startswith("initial_") or name == "post_refit_public_bc":
            row["epochs"] = epochs
            if name == "post_refit_public_bc":
                row["warmup_masked_kl_weight"] = p.REGISTERED_BC_SPEC[
                    "post_refit_warmup_masked_kl_weight"]
        else:
            row["fit"] = {
                "epochs": epochs,
                "batch_size": p.REGISTERED_BC_SPEC["batch_size"],
                "risk_weight": p.REGISTERED_BC_SPEC["risk_weight"],
                "margin_gap": p.REGISTERED_BC_SPEC["margin_gap"],
                "margin_weight": (0.0 if name == "risk_public_bc_fit" else
                                  p.REGISTERED_BC_SPEC["margin_weight"]),
            }
        stages.append(row)
    result = {
        "bc_recipe_version": p.REGISTERED_BC_RECIPE_VERSION,
        "bc_recipe_sha256": p.REGISTERED_BC_SPEC_SHA256,
        "executable_recipe_file_sha256": p.REGISTERED_BC_SPEC[
            "executable_recipe_file_sha256"],
        "risk_dual_config_sha256": p.REGISTERED_RISK_DUAL_CONFIG_SHA256,
        "source_closure_sha256": "f" * 64,
        "feature_schema_sha256": p.REGISTERED_BC_SPEC["feature_schema_sha256"],
        "action_physics_config_sha256": p.REGISTERED_BC_SPEC[
            "action_physics_config_sha256"],
        "bc_data_seed_blocks": schedule,
        "bc_data_seed_schedule_sha256": p._hash_json(schedule),
        "bc_critic_reset_schedule": critic_reset,
        "bc_critic_reset_schedule_sha256": p._hash_json(critic_reset),
        "teacher_uses_hidden_goal_or_route_state": False,
        "offline_critic_used_at_execution": False,
        "topology_calibration_eligibility_validation_used_for_bc_updates": False,
        "collision_critic": {
            "version": p.REGISTERED_BC_SPEC["collision_critic_version"],
            "epochs": p.REGISTERED_BC_SPEC["collision_critic_epochs"],
            "batch_size": p.REGISTERED_BC_SPEC["collision_critic_batch_size"],
            "learning_rate": p.REGISTERED_BC_SPEC[
                "collision_critic_learning_rate"],
            "positive_weight": p.REGISTERED_BC_SPEC[
                "collision_critic_positive_weight"],
            "seed": critic_reset["seed"],
            "environment_reset_seed": critic_reset["seed"],
            "environment_reset_count": critic_reset["environment_reset_count"],
            "purpose": "offline_BC_gradient_only_not_execution",
            "labels_source": "env.step_robot_collision_events_delta",
            "legal_actions_rewritten_by_environment": 0,
            "validation_passed": True,
        },
        "bc_stages": stages,
        "pretrained_role_sha256": actor_roles,
        "warmup_frozen_nn_actor": {
            "sha256": "a" * 64, "role_sha256": warmup_roles},
        "topology_source_nn_role_sha256": {"fit": warmup_roles},
        "topology_episode_parts": {
            name: {"scenario_seeds": seeds}
            for name, seeds in warmup_schedule.items()},
        "final_bc_internal_precheck": {
            "purpose": "development_internal_precheck_after_complete_bc",
            "passed": True,
            "final_bc_actor": {"sha256": "b" * 64,
                               "role_sha256": actor_roles},
            "action_physics_config_sha256": p.REGISTERED_BC_SPEC[
                "action_physics_config_sha256"],
            "warmup_actor_sha256": "a" * 64,
            "warmup_topology_fit_seed_schedule_sha256": p._hash_json(
                warmup_schedule),
            "final_bc_seed_schedule_sha256": p._hash_json(final_schedule),
            "final_bc_visits": {name: {"scenario_seeds": seeds}
                                for name, seeds in final_schedule.items()},
            "roles": {role: {"actor_role_sha256": actor_roles[role],
                             "refit_report": {"validation_passed": True}}
                      for role in p.ROLES},
        },
        "final_bc_internal_precheck_receipt": {
            "path": "/tmp/candidate/final-bc-precheck.json", "sha256": ""},
    }
    result["final_bc_internal_precheck_receipt"]["sha256"] = sha256(
        (json.dumps(result["final_bc_internal_precheck"],
                    sort_keys=True, allow_nan=False) + "\n").encode()).hexdigest()
    return result


def _write_json(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value, sort_keys=True,
                               separators=(",", ":"), allow_nan=False) + "\n")
    return sha256(path.read_bytes()).hexdigest()


def _stage1_pressure_fixture(root: Path) -> dict:
    """Synthetic receipts only; no environment is constructed or reset."""
    from experiments.warehouse import stage2_evidence as evidence
    from env.warehouse.public_action_features_v614 import ACTION_PHYSICS_CONFIG

    actor_path = root / "final-bc-dual-actor.npz"
    actor_path.write_bytes(b"static frozen actor fixture")
    actor_sha = evidence._sha(actor_path)
    roles = {role: sha256(role.encode()).hexdigest() for role in p.ROLES}
    script = root / "lock_scripts" / "full_bc12469.py"
    script.parent.mkdir()
    script.write_text("# static script fixture\n")
    source = root / "source"
    helper = source / "experiments/warehouse/quality_replay.py"
    helper.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "experiments/warehouse/quality_replay.py", helper)
    source_recipe = source / "configs/warehouse_v614_exploratory_recipe.json"
    source_recipe.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "configs/warehouse_v614_exploratory_recipe.json",
                    source_recipe)
    source_files = {"experiments/warehouse/quality_replay.py":
                    evidence._sha(helper),
                    "configs/warehouse_v614_exploratory_recipe.json":
                    evidence._sha(source_recipe)}
    source_sha = evidence._digest(source_files)
    snapshot_sha = _write_json(root / "source-snapshot.json", {
        "schema": "warehouse-v614-source-snapshot.v1",
        "source_files_sha256": source_files,
        "source_closure_sha256": source_sha})
    shutil.copyfile(source_recipe, root / "training-recipe.json")
    recipe_sha = evidence._sha(root / "training-recipe.json")
    quality_schedule = evidence._v614_schedule(
        [{"task": task, "base_seed": base, "stream": 0,
          "scenario_seed": base * 10 + task,
          "action_seed": base * 10 + task, "seed": base * 10 + task}
         for task in p.TASKS for base in range(1240500, 1240532)],
        first=1240500, count=32, streams=range(1))
    pressure_schedule = evidence._v614_schedule(
        [{"task": task, "base_seed": base, "stream": stream,
          "scenario_seed": base * 10 + task,
          "action_seed": base * 10 + task + stream * 1_000_000,
          "seed": base * 10 + task}
         for task in p.TASKS for base in range(1240200, 1240232)
         for stream in range(3)],
        first=1240200, count=32, streams=range(3))
    bc_schedule = p.bc_data_seed_schedule(12469)
    critic = p.bc_critic_reset_schedule(12469, p.INITIAL_RELEASE_ID)
    sequence = []
    for stage, epochs in p.bc_stage_receipt_shape():
        if epochs is None:
            sequence.extend({"phase": "bc", "seed": seed}
                            for seed in bc_schedule[stage])
            if stage == "topology_fit_argmax":
                sequence.extend({"phase": "bc", "seed": critic["seed"]}
                                for _ in range(2))
    for phase, schedule in (("quality", quality_schedule),
                            ("pressure", pressure_schedule)):
        sequence.extend({"phase": phase, "seed": row["seed"]}
                        for row in schedule)
        sequence.extend({"phase": phase, "seed": row["seed"]}
                        for row in schedule)
    program_schedule = {}
    first = 1240300
    for purpose, count in (("fit", 50), ("calibration", 25),
                           ("eligibility", 25), ("validation", 24)):
        program_schedule[purpose] = [
            {"task": task, "base_seed": base, "stream": 0,
             "scenario_seed": base * 10 + task,
             "action_seed": base * 10 + task, "seed": base * 10 + task}
            for task in p.TASKS for base in range(first, first + count)]
        first += count
    program_rows = [{"phase": "program", "seed": row["seed"]}
                    for purpose in program_schedule
                    for row in program_schedule[purpose]]
    external_sequence = sequence[1396:] + program_rows * 2
    journal_rows = [{**row, "episode_id": 1,
                     "map_layout_id": ACTION_PHYSICS_CONFIG["map_layout_id"],
                     "move_battery_cost": ACTION_PHYSICS_CONFIG[
                         "move_battery_cost"],
                     "charge_per_wait": ACTION_PHYSICS_CONFIG["charge_per_wait"]}
                    for row in sequence[:2164]]
    journal = root / "reset-audit.jsonl"
    journal.write_text("".join(json.dumps(row, sort_keys=True,
                                          separators=(",", ":")) + "\n"
                               for row in journal_rows))
    quality_journal_sha = sha256(b"\n".join(
        journal.read_bytes().splitlines()[:1588]) + b"\n").hexdigest()
    common = {"candidate_id": "v614-static-candidate", "pair_seed": 12469,
              "source_closure_sha256": source_sha,
              "action_physics_config_sha256": p.REGISTERED_BC_SPEC[
                  "action_physics_config_sha256"]}
    pre = {**common, "schema": "warehouse-v614-12469-predeclared-run.v1",
           "source_snapshot_sha256": snapshot_sha,
           "training_recipe_file_sha256": recipe_sha,
           "lock_script_path": str(script),
           "lock_script_sha256": evidence._sha(script),
           "required_cwd": str(root),
           "bc_recipe_sha256": p.REGISTERED_BC_SPEC_SHA256,
           "bc_data_seed_schedule": bc_schedule,
           "bc_data_seed_schedule_sha256": evidence._digest(bc_schedule),
           "bc_critic_reset_schedule": critic,
           "bc_critic_reset_schedule_sha256": evidence._digest(critic),
           "first_quality_schedule": quality_schedule,
           "quality_schedule_sha256": evidence._digest(quality_schedule),
           "reserved_pressure_schedule": pressure_schedule,
           "pressure_schedule_sha256": evidence._digest(pressure_schedule),
           "reserved_post_bc_program_schedule": program_schedule,
           "program_schedule_sha256": evidence._digest(program_schedule),
           "reserved_ppo_final_program_schedule": {"fit": []},
           "stage1_expected_reset_sequence": external_sequence,
           "stage1_expected_reset_sequence_sha256": evidence._digest(external_sequence),
           "first_quality_screen": {"mean_deliveries_min": 5.5,
                                    "mean_collisions_max": 6.0,
                                    "shutdown_episode_rate_max": 0.05},
           "pressure_screen": {"mean_deliveries_min": 5.5,
                               "mean_collisions_max": 6.0,
                               "shutdown_episode_rate_max": 0.05}}
    pre["run_spec"] = {
        "candidate_id": common["candidate_id"], "pair_seed": 12469,
        "bc_data_seed_schedule": bc_schedule,
        "bc_critic_reset_schedule": critic,
        "source_closure_sha256": source_sha,
        "stage1_expected_reset_sequence": external_sequence,
        "reset_audit_journal_path": str(root / "reset-audit.jsonl")}
    pre_sha = _write_json(root / "predeclared-run.json", pre)
    authorization = {**common,
                     "authorization_kind": "warehouse-v614-stage1-locked-run-v1",
                     "lock_script_path": str(script),
                     "lock_script_sha256": evidence._sha(script),
                     "bc_recipe_sha256": p.REGISTERED_BC_SPEC_SHA256,
                     "executable_recipe_file_sha256": recipe_sha,
                     "bc_data_seed_schedule_sha256": evidence._digest(bc_schedule),
                     "run_spec_sha256": evidence._digest(pre["run_spec"]),
                     "stage1_expected_reset_sequence_sha256":
                         evidence._digest(external_sequence)}
    auth_sha = _write_json(root / "stage1-authorization.json", authorization)
    independent_sha = _write_json(root / "seed-inventory-independent.json", {
        "all_candidate_groups_disjoint_from_protected": True,
        "all_candidate_groups_pairwise_disjoint": True})
    history_sha = _write_json(root / "historical-seed-audit.json", {
        "schema": "warehouse-v614-12469-historical-seed-audit.v1",
        "candidate_pair_seed": 12469,
        "candidate_schedule_sha256": evidence._digest({
            "bc": bc_schedule, "critic": critic,
            "quality": quality_schedule, "pressure": pressure_schedule,
            "program": program_schedule,
            "ppo_final_program": pre["reserved_ppo_final_program_schedule"]}),
        "candidate_historical_overlap_count": 0,
        "candidate_registered_sealed_overlap_count": 0,
        "candidate_internal_overlap_count": 0,
        "independent_auditor_receipt_sha256": independent_sha})
    lock = {**common, "schema": "warehouse-v614-12469-script-lock.v1",
            "predeclared_sha256": pre_sha,
            "source_snapshot_sha256": snapshot_sha,
            "training_recipe_file_sha256": recipe_sha,
            "lock_script_path": str(script),
            "lock_script_sha256": evidence._sha(script),
            "required_cwd": str(root),
            "python_command": f"python3 -I -B {script}",
            "authorization_receipt_sha256": auth_sha,
            "historical_seed_audit_sha256": history_sha,
            "independent_seed_inventory_sha256": independent_sha,
            "stage1_expected_reset_sequence_sha256": evidence._digest(external_sequence)}
    lock_sha = _write_json(root / "script-lock.json", lock)
    bc = _receipt()
    bc["source_closure_sha256"] = source_sha
    bc["final_bc_internal_precheck"]["final_bc_actor"]["sha256"] = actor_sha
    internal_path = root / "final-bc-internal-precheck.json"
    internal_path.write_text(json.dumps(bc["final_bc_internal_precheck"],
                                        sort_keys=True, allow_nan=False) + "\n")
    bc["final_bc_internal_precheck_receipt"]["path"] = str(internal_path)
    bc["final_bc_internal_precheck_receipt"]["sha256"] = sha256(
        internal_path.read_bytes()).hexdigest()
    bc_sha = _write_json(root / "bc-training-receipt.json", {
        "bc_report": bc, "final_bc_actor_path": str(actor_path),
        "final_bc_actor_sha256": actor_sha,
        "final_bc_role_sha256": roles})
    def result(label: str, schedule: list[dict], journal_sha: str,
               journal_count: int) -> dict:
        from experiments.warehouse.quality_replay import (
            ACTION_ENCODING, REPLAY_SCHEMA, TRACE_SCHEMA, claimed_series,
        )

        raw_path, replay_path = (root / f"{label}-episodes.jsonl",
                                 root / f"{label}-replay.jsonl")
        chain = "a" * 64
        raw = [{**row, "deliveries": 6, "collisions": 0,
                "shutdown_episode": False,
                "quality_trace": {
                    "schema": TRACE_SCHEMA, "actor_sha256": actor_sha,
                    "actor_role_sha256": roles,
                    "scenario_seed": row["seed"],
                    "action_seed": row["action_seed"],
                    "episode_id": 1, "frame_count": 1,
                    "action_encoding": ACTION_ENCODING,
                    "actions_b64": "AA==",
                    "initial_pre_state_sha256": "b" * 64,
                    "final_post_state_sha256": "c" * 64,
                    "chain_sha256": chain}}
               for row in schedule]
        replay = [{**row, "deliveries": 6, "collisions": 0,
                   "shutdown_episode": False, "chain_sha256": chain,
                   "independent_physical_replay": True,
                   "replay_source_sha256": evidence._sha(helper)}
                  for row in schedule]
        raw_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                                    for row in raw))
        replay_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                                       for row in replay))
        claimed = claimed_series(
            actor_sha256=actor_sha, role_sha256=roles, rows=raw,
            seeds=[(row["seed"], row["action_seed"]) for row in schedule])
        assert claimed["schema"] == REPLAY_SCHEMA
        replay_receipt_sha = _write_json(
            root / f"{label}-replay-receipt.json",
            {**claimed, "independent_physical_replay": True,
             "replay_source_sha256": evidence._sha(helper),
             "raw_jsonl_sha256": evidence._sha(raw_path),
             "replay_jsonl_sha256": evidence._sha(replay_path)})
        return {**common, "schema": f"warehouse-v614-stage1-{label}-result.v1",
                "status": f"unregistered_frozen_bc_{label}_complete",
                "predeclared_sha256": pre_sha,
                "script_lock_sha256": lock_sha,
                "authorization_receipt_sha256": auth_sha,
                "runner_script_sha256": evidence._sha(script),
                "training_recipe_sha256": recipe_sha,
                "actor_sha256": actor_sha, "actor_role_sha256": roles,
                "bc_training_receipt_sha256": bc_sha,
                f"{label}_schedule_sha256": evidence._digest(schedule),
                f"{label}_episode_count": len(schedule),
                "raw_jsonl_sha256": evidence._sha(raw_path),
                "replay_jsonl_sha256": evidence._sha(replay_path),
                "physical_replay_receipt_sha256": replay_receipt_sha,
                "replay_source_sha256": evidence._sha(helper),
                "reset_journal_prefix_sha256": journal_sha,
                "reset_journal_prefix_row_count": journal_count,
                "reset_schedule_prefix_sha256": evidence._digest(
                    external_sequence[:192 if label == "quality" else 768]),
                f"{label}_measured": {"mean_deliveries": 6.0,
                                      "mean_collisions": 0.0,
                                      "shutdown_episode_rate": 0.0},
                f"{label}_gates": {"mean_deliveries": True,
                                   "mean_collisions": True,
                                   "shutdown_episode_rate": True},
                f"{label}_passed": True,
                "program_or_ppo_used_at_pressure": False,
                "registered_or_sealed_seeds_used": False}

    quality = result("quality", quality_schedule, quality_journal_sha, 1588)
    quality_sha = _write_json(root / "quality-result.json", quality)
    pressure = result("pressure", pressure_schedule, evidence._sha(journal),
                      2164)
    pressure.update({"status": "unregistered_frozen_bc_pressure_complete",
                     "quality_result_sha256": quality_sha,
                     "pressure_episode_count": 288,
                     "registered_or_sealed_seeds_used": False,
                     "program_or_ppo_used_at_pressure": False})
    _write_json(root / "pressure-result.json", pressure)
    return {"expected_candidate_id": common["candidate_id"],
            "expected_pair_seed": 12469,
            "expected_actor_sha256": actor_sha,
            "expected_role_sha256": roles,
            "expected_source_closure_sha256": source_sha,
            "expected_training_recipe_sha256": recipe_sha,
            "expected_quality_schedule_sha256": evidence._digest(quality_schedule),
            "expected_pressure_schedule_sha256": evidence._digest(pressure_schedule)}


class WarehouseV614StaticProtocolTest(unittest.TestCase):
    def test_stage1_pressure_rejects_journal_source_and_program_plan_drift(self):
        from experiments.warehouse.stage2_evidence import (
            verify_stage1_pressure_receipt_identity,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            expected = _stage1_pressure_fixture(root)
            verify_stage1_pressure_receipt_identity(root, **expected)
            journal_path = root / "reset-audit.jsonl"
            original_journal = journal_path.read_bytes()
            journal_path.write_bytes(original_journal + original_journal.splitlines()[-1] + b"\n")
            with self.assertRaisesRegex(ValueError, "row count"):
                verify_stage1_pressure_receipt_identity(root, **expected)
            journal_path.write_bytes(original_journal)
            rows = [json.loads(line) for line in original_journal.splitlines()]
            rows[-1]["seed"] += 1
            journal_path.write_text("".join(json.dumps(row, sort_keys=True,
                                                  separators=(",", ":")) + "\n"
                                           for row in rows))
            result_path = root / "pressure-result.json"
            original_result = result_path.read_bytes()
            changed_result = json.loads(original_result)
            changed_result["reset_journal_prefix_sha256"] = sha256(
                journal_path.read_bytes()).hexdigest()
            _write_json(result_path, changed_result)
            with self.assertRaisesRegex(ValueError, "reset journal prefix differs"):
                verify_stage1_pressure_receipt_identity(root, **expected)
            result_path.write_bytes(original_result)
            journal_path.write_bytes(original_journal)
            extra = root / "source" / "undeclared.py"
            extra.write_text("# extra source\n")
            with self.assertRaisesRegex(ValueError, "source snapshot inventory"):
                verify_stage1_pressure_receipt_identity(root, **expected)
            extra.unlink()
            helper = root / "source/experiments/warehouse/quality_replay.py"
            helper_original = helper.read_bytes()
            helper.write_bytes(helper_original + b"# drift\n")
            with self.assertRaisesRegex(ValueError, "source snapshot bytes"):
                verify_stage1_pressure_receipt_identity(root, **expected)
            helper.write_bytes(helper_original)
            pre_path = root / "predeclared-run.json"
            pre_original = pre_path.read_bytes()
            pre = json.loads(pre_original)
            pre["reserved_post_bc_program_schedule"]["fit"][0]["seed"] += 1
            _write_json(pre_path, pre)
            with self.assertRaises(ValueError):
                verify_stage1_pressure_receipt_identity(root, **expected)
            pre_path.write_bytes(pre_original)
            verify_stage1_pressure_receipt_identity(root, **expected)

    def test_single_script_pressure_identity_rejects_forgery_without_reset(self):
        from experiments.warehouse.stage2_evidence import (
            verify_stage1_pressure_receipt_identity,
        )
        from env.warehouse.environment import WarehouseMultiAgentEnv

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            expected = _stage1_pressure_fixture(root)
            with patch.object(WarehouseMultiAgentEnv, "reset",
                              side_effect=AssertionError("no reset authorized")):
                checked = verify_stage1_pressure_receipt_identity(
                    root, **expected)
            self.assertTrue(checked["independent_physical_replay_equal"])
            result_path = root / "pressure-result.json"
            original = json.loads(result_path.read_text())
            for change in (
                {"schema": "warehouse-v69-pressure-result.v1"},
                {"actor_sha256": "0" * 64},
                {"quality_result_sha256": "0" * 64},
                {"reset_schedule_prefix_sha256": "0" * 64},
            ):
                _write_json(result_path, {**original, **change})
                with self.assertRaises(ValueError):
                    verify_stage1_pressure_receipt_identity(root, **expected)
            _write_json(result_path, original)
            replay_path = root / "pressure-replay.jsonl"
            replay_original = replay_path.read_bytes()
            rows = [json.loads(line) for line in replay_original.splitlines()]
            rows[0]["independent_physical_replay"] = False
            replay_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                                           for row in rows))
            _write_json(result_path, {**original,
                       "replay_jsonl_sha256": sha256(
                           replay_path.read_bytes()).hexdigest()})
            with self.assertRaisesRegex(ValueError, "physical replay differs"):
                verify_stage1_pressure_receipt_identity(root, **expected)
            replay_path.write_bytes(replay_original)
            _write_json(result_path, original)
            internal_path = root / "final-bc-internal-precheck.json"
            internal_original = internal_path.read_bytes()
            internal_path.write_bytes(internal_original + b" ")
            with self.assertRaisesRegex(ValueError, "final BC precheck bytes"):
                verify_stage1_pressure_receipt_identity(root, **expected)
            internal_path.write_bytes(internal_original)
            actor_path = root / "final-bc-dual-actor.npz"
            actor_path.write_bytes(b"forged different Actor")
            with self.assertRaisesRegex(ValueError, "frozen BC Actor bytes"):
                verify_stage1_pressure_receipt_identity(root, **expected)

    def test_executable_runner_matches_protocol_without_reset(self):
        from backend.training import warehouse_v6_runner as runner
        from backend.training.warehouse_v6_actor import MODEL_VERSION

        self.assertEqual(runner.BC_RECIPE_VERSION,
                         p.REGISTERED_BC_RECIPE_VERSION)
        self.assertEqual(MODEL_VERSION,
                         p.REGISTERED_BC_SPEC["actor_model_version"])
        self.assertEqual(runner.validate_v614_exploratory_recipe(),
                         p.REGISTERED_BC_SPEC["executable_recipe_file_sha256"])
        recipe = runner.v614_executable_recipe()
        self.assertEqual(recipe["initial_teacher_episodes"],
                         p.REGISTERED_BC_SPEC["bc_episodes"])
        self.assertEqual(recipe["initial_bc_epochs"],
                         p.REGISTERED_BC_SPEC["bc_epochs"])
        self.assertEqual(recipe["initial_dagger_episodes"],
                         p.REGISTERED_BC_SPEC["dagger_episodes"])
        self.assertEqual(recipe["gate_supervision_weight"],
                         p.REGISTERED_BC_SPEC["initial_gate_loss_weight"])
        self.assertEqual(recipe["post_refit_bc_epochs"],
                         p.REGISTERED_BC_SPEC["post_refit_bc_epochs"])
        self.assertEqual(recipe["post_refit_warmup_masked_kl_weight"],
                         p.REGISTERED_BC_SPEC["post_refit_warmup_masked_kl_weight"])
        self.assertEqual(recipe["feature_schema_sha256"],
                         p.REGISTERED_BC_SPEC["feature_schema_sha256"])
        self.assertEqual(recipe["action_physics_config_sha256"],
                         p.REGISTERED_BC_SPEC["action_physics_config_sha256"])
        for prefix, counts in (("topology_", runner.BC_TOPOLOGY_STAGE_COUNTS),
                               ("final_bc_precheck_",
                                runner.BC_FINAL_CORE_STAGE_COUNTS)):
            for name, (stage, count) in counts.items():
                self.assertEqual(
                    p.bc_data_seed_schedule(12469)[prefix + name],
                    runner._bc_stage_seeds(12469, stage, count))

    def test_exact_bc_stage_seed_inventory_and_candidate_isolation(self):
        schedule = p.bc_data_seed_schedule(12469)
        seeds = [seed for values in schedule.values() for seed in values]
        self.assertEqual((len(schedule), len(seeds), len(set(seeds))),
                         (26, 1394, 1394))
        self.assertEqual(p._hash_json(sorted(seeds)),
                         "d674c5040b0192efdf951073b3241425dec8241c547923159b855b5b5c3314ef")
        self.assertEqual(p.bc_critic_reset_schedule(
            12469, p.INITIAL_RELEASE_ID),
            {"seed": 18669, "environment_reset_count": 2})
        with self.assertRaises(ValueError):
            p.PairRunSpec("candidate", 12469, 200_000, "/tmp/out",
                          "actor_sparse_topology_v1",
                          checkpoint_seeds=p.DEVELOPMENT_SELECTION_SEEDS)

    def test_internal_purposes_and_external_stage2_are_disjoint(self):
        schedule = p.bc_data_seed_schedule(12469)
        warmup = set(schedule["topology_fit_sampled"] +
                     schedule["topology_fit_argmax"])
        four = {purpose: set().union(*(
            set(schedule[f"final_bc_precheck_{purpose}_{kind}"])
            for kind in ("sampled", "argmax")))
            for purpose in ("fit", "calibration", "eligibility", "validation")}
        external = {base * 10 + task
                    for base in range(1240300, 1240424)
                    for task in p.TASKS}
        self.assertEqual([len(four[name]) for name in four], [50, 25, 25, 24])
        self.assertEqual(len(external), 372)
        for index, a in enumerate(four):
            self.assertFalse(warmup & four[a])
            self.assertFalse(external & four[a])
            for b in list(four)[index + 1:]:
                self.assertFalse(four[a] & four[b])

    def test_recipe_file_and_gate_thresholds_remain_fixed(self):
        from env.warehouse.domain import participant_study_config
        from env.warehouse.public_action_features_v614 import (
            validate_study_action_physics,
        )

        recipe = ROOT / "configs/warehouse_v614_exploratory_recipe.json"
        self.assertEqual(sha256(recipe.read_bytes()).hexdigest(),
                         p.REGISTERED_BC_SPEC["executable_recipe_file_sha256"])
        self.assertEqual(validate_study_action_physics(
            participant_study_config()),
            p.REGISTERED_BC_SPEC["action_physics_config_sha256"])
        with self.assertRaises(ValueError):
            validate_study_action_physics(participant_study_config(
                move_battery_cost=3))
        self.assertTrue(p.quality_pass({"mean_deliveries": 5.5,
                                        "mean_collisions": 6,
                                        "shutdown_episode_rate": 0.05},
                                       development=True))
        self.assertFalse(p.quality_pass({"mean_deliveries": 5.49,
                                         "mean_collisions": 6,
                                         "shutdown_episode_rate": 0.05},
                                        development=True))
        self.assertTrue(p.quality_pass({"mean_deliveries": 5,
                                        "mean_collisions": 8,
                                        "shutdown_episode_rate": 0.10},
                                       development=False))
        self.assertEqual(p.PROGRAM_LAMBDA, .01)
        self.assertEqual((p.PROGRAM_START_FRACTION, p.PROGRAM_RAMP_FRACTION),
                         (.80, .05))
        for development, minimum_coverage, feedback, explanation, maximum_kl in (
                (True, .30, .93, .95, .25),
                (False, .20, .85, .90, .35)):
            for purpose, agreement in (("feedback", feedback),
                                       ("explanation", explanation)):
                row = {"coverage": minimum_coverage,
                       "actionable_coverage": minimum_coverage,
                       "agreement": agreement, "kl": maximum_kl,
                       "rows": 100, "actionable_rows": 20,
                       "illegal_count": 0}
                self.assertTrue(p.program_pass(row, purpose=purpose,
                                               development=development))
                for key, worse in (("coverage", minimum_coverage - .001),
                                   ("agreement", agreement - .001),
                                   ("kl", maximum_kl + .001),
                                   ("illegal_count", 1)):
                    modified = {**row, key: worse}
                    self.assertFalse(p.program_pass(
                        modified, purpose=purpose, development=development))

    def test_receipt_rejects_stage_and_actor_binding_tampering(self):
        base = _receipt()
        self.assertTrue(p.bc_report_valid(base, 12469, "f" * 64))
        for mutate in (
            lambda r: r["bc_stages"][-1].update(
                {"episode_seeds": r["bc_stages"][-2]["episode_seeds"]}),
            lambda r: r["final_bc_internal_precheck"]["final_bc_visits"][
                "calibration_sampled"].update({"scenario_seeds": [1]}),
            lambda r: r["topology_episode_parts"]["fit_sampled"].update(
                {"scenario_seeds": [1]}),
            lambda r: r["final_bc_internal_precheck"]["roles"]["robot_1"].update(
                {"actor_role_sha256": "0" * 64}),
            lambda r: r["final_bc_internal_precheck_receipt"].update(
                {"sha256": "0" * 64}),
            lambda r: r["final_bc_internal_precheck_receipt"].update(
                {"path": ""}),
            lambda r: r.update({"executable_recipe_file_sha256": "0" * 64}),
            lambda r: r.update({"topology_calibration_eligibility_validation_used_for_bc_updates": True}),
        ):
            changed = deepcopy(base)
            mutate(changed)
            self.assertFalse(p.bc_report_valid(changed, 12469, "f" * 64))

    def test_packager_rejects_actor_with_different_physics_config(self):
        from scripts import package_warehouse_v6_release as package

        receipt = {"selected_pair_seed": 6101, "locked_budget": 200_000,
                   "candidate_id": "candidate",
                   "action_physics_config_sha256": p.REGISTERED_BC_SPEC[
                       "action_physics_config_sha256"]}
        pair = {"candidate_id": "candidate", "pair_seed": 6101,
                "joint_steps": 200_000,
                "variants": {name: {"actor_artifact": {
                    "action_physics_config_sha256": "0" * 64}}
                    for name in p.VARIANTS}}
        with patch.object(package, "_read", return_value=pair), \
                patch.object(package, "_bundle", side_effect=AssertionError(
                    "no asset should be opened before physics identity")):
            with self.assertRaisesRegex(ValueError, "physics config"):
                package._policies(receipt,
                                  {"pair_reports": {"6101": {"path": "unused"}}},
                                  Path("/tmp"), {})

    def test_release_catalog_rejects_missing_physics_hash(self):
        from backend.training.warehouse_v6_actor import FEATURE_SCHEMA_SHA256, MODEL_VERSION
        from study_v3 import release_catalog as catalog

        release_id = catalog.RELEASE_ID
        spec = catalog.warehouse_release_spec(release_id)
        roles = {"human": "a" * 64, "ai": "b" * 64}
        entry = {"model_version": MODEL_VERSION,
                 "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
                 "actor_path": "unused", "actor_sha256": "a" * 64,
                 "training_report_path": "unused",
                 "training_report_sha256": "a" * 64,
                 "acceptance_path": "unused", "acceptance_sha256": "a" * 64,
                 "role_sha256": roles, "program_paths": [],
                 "program_sha256": {}}
        entries = {name: {**entry, "policy_id": spec.policy_ids[name]}
                   for name in p.VARIANTS}
        catalog_data = {"warehouse": {"status": "accepted",
                                      "policy_ids": spec.policy_ids},
                        "policies": {"warehouse": entries}}
        with patch.object(catalog, "_catalog", return_value=catalog_data), \
                patch.object(catalog, "_asset", side_effect=AssertionError(
                    "no asset should be opened before physics identity")):
            with self.assertRaisesRegex(catalog.ReleaseAssetUnavailable,
                                        "warehouse_policy_identity_mismatch"):
                catalog.warehouse_policy_artifacts(release_id)


if __name__ == "__main__":
    unittest.main()
