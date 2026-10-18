"""Synthetic Warehouse v6.16 receipts for protocol tests only.

No environment is constructed or reset by this fixture.
"""

from hashlib import sha256
import json
import numpy as np

from experiments.warehouse import protocol as p


def topology_freeze_evidence(joint_steps: int) -> dict:
    """Synthetic v6.16 proof that both role cores stayed certified at freeze."""
    frozen = {"robot_1": "a" * 64, "robot_2": "b" * 64}
    actors = {"robot_1": "c" * 64, "robot_2": "d" * 64}
    leaves = {"robot_1": "e" * 64, "robot_2": "f" * 64}
    core_mask = np.zeros(p.CORE_MASK_LEAVES, dtype=np.bool_)
    core_mask[0] = True
    qualification_mask_sha = sha256(core_mask.tobytes()).hexdigest()
    installed_mask_sha = sha256(core_mask.astype(np.float32).tobytes()).hexdigest()
    qualification = {
        "stage": f"ppo_freeze_{p.topology_frozen_at_step(joint_steps)}",
        "role_tree_sha256": frozen,
        "route_parity_passed": True,
        "path_legality_passed": True,
        "core_qualification_recomputed": True,
        "roles": {role: {
            "topology_sha256": frozen[role],
            "qualification_bound_to_installed_actor": True,
            "qualification_failed_core_reset": False,
            "qualification": {
                "method": p.REGISTERED_ACTOR_CORE_QUALIFICATION_METHOD,
                "validation_passed": True,
                "frozen_role_sha256": actors[role],
                "actor_topology_sha256": frozen[role],
                "core_mask_sha256": qualification_mask_sha,
                "selected_core_leaf_ids": [0],
                "leaf_path_proof": {"0": {
                    "raw_top": "WAIT", "proven_actions": ["WAIT"],
                    "candidate": True, "selected": True,
                }},
            },
            "installed_core": {"core_leaf_count": 1,
                               "core_mask_sha256": installed_mask_sha},
            "parity": {"routing_mismatches": 0},
        } for role in p.ROLES},
    }
    return {
        "topology_frozen_at_step": p.topology_frozen_at_step(joint_steps),
        "frozen_topology_role_sha256": frozen,
        "final_topology_role_sha256": dict(frozen),
        "frozen_core_leaf_logits_sha256": leaves,
        "final_core_leaf_logits_sha256": dict(leaves),
        "frozen_actor_role_sha256": actors,
        "topology_freeze_qualification": qualification,
        "topology_updates_after_freeze": 0,
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
        "bc_topology_refresh_history": [{
            "stage": stage,
            "role_tree_sha256": {role: sha256(
                f"{role}:{stage}:tree".encode()).hexdigest() for role in p.ROLES},
            "route_parity_passed": True,
            "path_legality_passed": True,
            "core_qualification_recomputed": True,
            "roles": {role: {
                "topology_sha256": sha256(
                    f"{role}:{stage}:tree".encode()).hexdigest(),
                "training_scene_count": 100,
                "qualification": {"method":
                    p.REGISTERED_ACTOR_CORE_QUALIFICATION_METHOD},
            } for role in p.ROLES},
        } for stage in p.bc_topology_refresh_stages()],
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
            "final_bc_visits": {name: {"scenario_seeds": seeds}
                                for name, seeds in final_schedule.items()},
            "roles": {role: {"actor_role_sha256": actor_roles[role],
                             "refit_report": {"validation_passed": True},
                             "actor_core_qualification": {
                                 "method": p.REGISTERED_ACTOR_CORE_QUALIFICATION_METHOD,
                                 "validation_passed": True},
                             "installed_core_validation_mass": 0.40,
                             "installed_core_expected_actionable_mass": 0.35}
                      for role in p.ROLES},
        },
        "final_bc_internal_precheck_receipt": {
            "path": "/tmp/candidate/final-bc-precheck.json", "sha256": ""},
    }
    result["final_bc_internal_precheck_receipt"]["sha256"] = sha256(
        (json.dumps(result["final_bc_internal_precheck"],
                    sort_keys=True, allow_nan=False) + "\n").encode()).hexdigest()
    return result
