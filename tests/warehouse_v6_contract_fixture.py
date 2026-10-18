"""Synthetic Warehouse active-contract receipts for protocol tests only.

No environment is constructed or reset by this fixture.
"""

from hashlib import sha256
import json

from experiments.warehouse import protocol as p


def topology_diagnostic(*, stage: str | None = None,
                        joint_step: int | None = None,
                        support_locked: bool = True) -> dict:
    """Observe frozen arrays without rebuilding or qualifying the Actor."""
    trees = {"robot_1": "a" * 64, "robot_2": "b" * 64}
    actors = {"robot_1": "c" * 64, "robot_2": "d" * 64}
    result = {
        "role_tree_sha256": trees,
        "role_support_sha256": {"robot_1": "e" * 64, "robot_2": "f" * 64},
        "actor_role_sha256_before": actors,
        "actor_role_sha256_after": dict(actors),
        "route_parity_passed": True,
        "guard_invariants_passed": True,
        "core_qualification_recomputed": False,
        "actor_rebuilt": False,
        "auxiliary_ce_updates": 0,
        "support_locked": support_locked,
        "roles": {role: {
            "topology_sha256": trees[role],
            "training_scene_count": 100,
            "parity": {"routing_mismatches": 0,
                       "maximum_logit_abs_difference": 0.0,
                       "role_sha256": actors[role]},
        } for role in p.ROLES},
    }
    if stage is not None:
        result["stage"] = stage
    if joint_step is not None:
        result["joint_step"] = joint_step
    return result


def topology_freeze_evidence(joint_steps: int) -> dict:
    diagnostic = topology_diagnostic(
        stage=f"ppo_freeze_{p.topology_frozen_at_step(joint_steps)}")
    frozen = diagnostic["role_tree_sha256"]
    actors = diagnostic["actor_role_sha256_after"]
    supports = {"robot_1": "e" * 64, "robot_2": "f" * 64}
    return {
        "topology_frozen_at_step": p.topology_frozen_at_step(joint_steps),
        "frozen_topology_role_sha256": frozen,
        "final_topology_role_sha256": dict(frozen),
        "frozen_support_role_sha256": supports,
        "final_support_role_sha256": dict(supports),
        "frozen_actor_role_sha256": actors,
        "topology_freeze_qualification": diagnostic,
        "topology_updates_after_freeze": 0,
        "core_leaf_freezing_used": False,
        "leaf_outputs_remain_trainable": True,
    }


def bc_receipt(pair_seed: int, source_closure_sha256: str) -> dict:
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
        "source_closure_sha256": source_closure_sha256,
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
        "bc_topology_refresh_history": [topology_diagnostic(
            stage=stage, support_locked=index == len(p.bc_topology_refresh_stages())-1)
            for index, stage in enumerate(p.bc_topology_refresh_stages())],
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
                               "model_version": p.REGISTERED_BC_SPEC[
                                   "actor_model_version"],
                               "role_sha256": actor_roles},
            "action_physics_config_sha256": p.REGISTERED_BC_SPEC[
                "action_physics_config_sha256"],
            "warmup_actor_sha256": "a" * 64,
            "warmup_topology_fit_seed_schedule_sha256": p._hash_json(
                warmup_schedule),
            "final_bc_seed_schedule_sha256": p._hash_json(final_schedule),
            "final_bc_visits": {name: {"scenario_seeds": seeds, "rows": 10, "physical_frames": 10, "excluded_frames": 0}
                                for name, seeds in final_schedule.items()},
            "roles": {role: {"actor_role_sha256": actor_roles[role],
                             "refit_report": {"validation_passed": True,
                                 "validation": {purpose: {
                                     "whole_tree_raw_illegal_count": 0,
                                     "whole_tree_path_uncertified_count": 0,
                                     "path_uncertified_count": 0}
                                     for purpose in ("feedback", "explanation")}},
                             "validation": {purpose: {
                                 "whole_tree_raw_illegal_count": 0,
                                 "whole_tree_path_uncertified_count": 0,
                                 "path_uncertified_count": 0}
                                 for purpose in ("feedback", "explanation")},
                             "program_sha256": sha256((role+":program").encode()).hexdigest(),
                             "qualification_changes_actor": False,
                             "program_qualification": {
                                 "method": p.REGISTERED_ACTOR_CORE_QUALIFICATION_METHOD,
                                 "validation_passed": True,
                                 "frozen_role_sha256": actor_roles[role],
                                 "program_sha256": sha256((role+":program").encode()).hexdigest()},
                             "qualified_validation_mass": 0.40,
                             "expected_feedback_actionable_mass": 0.35}
                      for role in p.ROLES},
        },
        "final_bc_internal_precheck_receipt": {
            "path": "/tmp/candidate/final-bc-precheck.json", "sha256": ""},
    }
    seeds = list(p.DEVELOPMENT_SELECTION_SEEDS if pair_seed in p.DEVELOPMENT_PAIR_SEEDS
                 else p.FORMAL_SELECTION_SEEDS)
    result["quality_screen_seeds"] = seeds
    result["final_bc_internal_precheck"]["quality_screen_seeds"] = seeds
    result["final_bc_internal_precheck"]["quality_screen"] = {"summary": {
        "mean_deliveries": 6.0, "mean_collisions": 2.0,
        "shutdown_episode_rate": 0.0}}

    result["final_bc_internal_precheck_receipt"]["sha256"] = sha256(
        (json.dumps(result["final_bc_internal_precheck"],
                    sort_keys=True, allow_nan=False) + "\n").encode()).hexdigest()
    return result
