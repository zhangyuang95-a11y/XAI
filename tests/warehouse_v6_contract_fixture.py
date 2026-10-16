"""Synthetic complete Warehouse v6.14 BC receipts for protocol tests only.

No environment is constructed or reset by this fixture.
"""

from hashlib import sha256
import json

from experiments.warehouse import protocol as p


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
