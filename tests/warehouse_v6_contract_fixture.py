"""Minimal synthetic Warehouse BC receipt for pure protocol tests only."""

from experiments.warehouse import protocol as p


def bc_receipt(pair_seed: int, source_closure_sha256: str) -> dict:
    schedule = p.bc_data_seed_schedule(pair_seed)
    critic_resets = p.bc_critic_reset_schedule(pair_seed, p.INITIAL_RELEASE_ID)
    stages = []
    for name, epochs in p.bc_stage_receipt_shape():
        if epochs is None:
            stages.append({"stage": name, "episode_seeds": schedule[name],
                           "kind": ("public_teacher" if "teacher" in name
                                    else "dual_nn_visitation")})
        elif name.startswith("initial_"):
            stages.append({
                "stage": name, "epochs": epochs,
                **{field: p.REGISTERED_BC_SPEC[f"initial_{field}"]
                   for field in ("tree_loss_weight", "gate_loss_weight",
                                 "bound_loss_weight")},
                "learning_rate": p.REGISTERED_BC_SPEC[
                    "initial_bc_learning_rate"],
                "batch_size": p.REGISTERED_BC_SPEC["initial_bc_batch_size"],
                "gradient_clip": p.REGISTERED_BC_SPEC[
                    "initial_bc_gradient_clip"],
            })
        else:
            initial_risk = name == "risk_public_bc_fit"
            stages.append({"stage": name, "fit": {
                "epochs": epochs, "batch_size": p.REGISTERED_BC_SPEC["batch_size"],
                "learning_rate": p.REGISTERED_BC_SPEC[
                    "risk_bc_learning_rate"],
                "gradient_clip": p.REGISTERED_BC_SPEC[
                    "risk_bc_gradient_clip"],
                "risk_weight": p.REGISTERED_BC_SPEC["risk_weight"],
                "margin_gap": p.REGISTERED_BC_SPEC["margin_gap"],
                "margin_weight": (0.0 if initial_risk else
                                  p.REGISTERED_BC_SPEC["margin_weight"]),
                "near_increment": (0.0 if initial_risk else
                                   p.REGISTERED_BC_SPEC["near_increment"]),
                "low_return_increment": p.REGISTERED_BC_SPEC[
                    "low_return_increment"],
            }})
    return {
        "bc_recipe_version": p.REGISTERED_BC_RECIPE_VERSION,
        "bc_recipe_sha256": p.REGISTERED_BC_SPEC_SHA256,
        "risk_dual_config_sha256": p.REGISTERED_RISK_DUAL_CONFIG_SHA256,
        "source_closure_sha256": source_closure_sha256,
        "bc_data_seed_blocks": schedule,
        "bc_data_seed_schedule_sha256": p._hash_json(schedule),
        "bc_critic_reset_schedule": critic_resets,
        "bc_critic_reset_schedule_sha256": p._hash_json(critic_resets),
        "teacher_uses_hidden_goal_or_route_state": False,
        "offline_critic_used_at_execution": False,
        "collision_critic": {
            "version": p.REGISTERED_BC_SPEC["collision_critic_version"],
            "epochs": p.REGISTERED_BC_SPEC["collision_critic_epochs"],
            "batch_size": p.REGISTERED_BC_SPEC[
                "collision_critic_batch_size"],
            "learning_rate": p.REGISTERED_BC_SPEC[
                "collision_critic_learning_rate"],
            "positive_weight": p.REGISTERED_BC_SPEC[
                "collision_critic_positive_weight"],
            "seed": critic_resets["seed"],
            "environment_reset_seed": critic_resets["seed"],
            "environment_reset_count": critic_resets["environment_reset_count"],
            "validation_passed": True,
        },
        "bc_stages": stages,
    }
