"""Immutable Warehouse v6 seed blocks, acceptance gates, and one-use claims.

This module contains no policy or environment action logic.  A program can
provide training feedback and audited evidence, never an executed action.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import struct
from typing import Mapping, Sequence


INITIAL_RELEASE_ID = "policylens-three-domain-20261016.v6.0"
BUDGETS = (200_000, 300_000, 400_000)
DEVELOPMENT_PAIR_SEEDS = (6140, 6141, 6142)
FORMAL_PAIR_SEEDS = (6101, 6102, 6103)
# Candidate-only development seed. PairRunSpec and release acceptance continue
# to admit only the registered three development and three formal pairs.
EXPLORATORY_PAIR_SEEDS = (12469, 12470, 12471)
DEVELOPMENT_SELECTION_SEEDS = tuple(range(263_100, 263_124))
DEVELOPMENT_STRESS_BASE_SEEDS = tuple(range(263_200, 263_456))
DEVELOPMENT_PROGRAM_AUDIT_SEEDS = tuple(range(263_500, 263_524))
DEVELOPMENT_ACTION_STREAMS = (0, 1, 2)
FORMAL_SELECTION_SEEDS = tuple(range(240_048, 240_072))
FORMAL_PROGRAM_PREAUDIT_SEEDS = tuple(range(240_072, 240_096))
FINAL_PROGRAM_AUDIT_SEEDS = tuple(range(240_100, 240_124))
HELDOUT_BASE_SEEDS = tuple(range(250_000, 250_100))
_RELEASE_PATTERN = re.compile(
    r"policylens-three-domain-20261016\.v6\.(0|[1-9][0-9]*)\Z")
TASKS = (1, 2, 3)
ROLES = ("robot_1", "robot_2")
VARIANTS = ("raw", "regularized")
CORE_MASK_LEAVES = 16
PROGRAM_LAMBDA = 0.01
PROGRAM_START_FRACTION = 0.80
PROGRAM_RAMP_FRACTION = 0.05
SCENARIO_BLOCK_STEPS = 120
ONLINE_PROGRAM_EXTRACTION_PERCENTAGES = (75, 84, 93)
TOPOLOGY_REFRESH_PERCENTAGES = (24, 48, 72)
TOPOLOGY_FREEZE_PERCENTAGE = 75
FINAL_PROGRAM_SPLIT_COUNTS = {"fit": 50, "calibration": 25,
                              "eligibility": 25, "validation": 24}
REGISTERED_CANDIDATE_ID = "warehouse-v617-risk-protected-02"
REGISTERED_ARCHITECTURE_MODE = "risk_stratified_protected_tree_256_residual.v6.17-candidate"
# A candidate names its fitting implementation explicitly. This list is not
# a release selection: the selected method is sealed only after development
# quality and independent fidelity gates pass. The old axis fit is excluded.
REGISTERED_PROGRAM_FIT_METHOD = "actor_risk_stratified_protected_topology_v3"
REGISTERED_ACTOR_CORE_QUALIFICATION_METHOD = "actor_risk_protected_core_qualification_v3"
PROGRAM_FIT_METHOD_CANDIDATES = (REGISTERED_PROGRAM_FIT_METHOD,)
# The failed 11969 v2 screen does not retire its extractor algorithm: a new
# Actor may reuse that implementation. Its concrete frozen Actor and source
# identities must never be promoted as a new registered training result.
RETIRED_119_ACTOR_SHA256 = "c4080edeaff2b633fbb274db709a0685ce70692978e5fc710c22c8553792584f"
RETIRED_119_SOURCE_CLOSURE_SHA256 = frozenset({
    "8569cdf9dcca314d5ec738a62c962bbbc51ea5c3125b3f37677ebb0b6b330be9",
    "1430c59278e28a672c636682225bb94d9ecfae655ccb937c26f6cee0df14642e",
})
RETIRED_119_MODEL_VERSION = "warehouse-dual-actor.v6.9-exploratory"
REGISTERED_BC_RECIPE_VERSION = (
    "warehouse-public-dual-bc.v6.17-risk-protected.r2")
LEGACY_V614_BC_RECIPE_VERSION = (
    "warehouse-public-dual-bc.v6.14-public-affordance-exploratory")
LEGACY_V614_BC_SPEC_SHA256 = (
    "f328512e3a2bb888892bb4366240dce8fc99d855a7591c2f0c18fdd1b3f183a6")
LEGACY_V614_RECIPE_FILE_SHA256 = (
    "fc45da901a8267175b051ba31ea10cedf327033b48fb9cbcec575b6c617c8ccb")
# The v6.16 candidate's training identity. The executable recipe JSON is
# separately byte-hashed below, then compared with the runner's constants
# before Stage 1 can create an environment.
REGISTERED_BC_SPEC = {
    "bc_episodes": 100,
    "bc_epochs": 20,
    "dagger_rounds": 2,
    "dagger_episodes": 40,
    "dagger_epochs": 10,
    "risk_teacher_episodes": 40,
    "risk_visit_episodes": 80,
    "large_teacher_episodes": 200,
    "large_visit_episodes": 120,
    "deep_teacher_episodes": 200,
    "deep_dagger_rounds": 4,
    "deep_dagger_episodes": 100,
    "risk_epochs": 8,
    "deep_epochs": 6,
    "batch_size": 128,
    "risk_weight": 2.0,
    "margin_gap": 3.0,
    "margin_weight": 0.03,
    "near_increment": 2.0,
    "low_return_increment": 8.0,
    "sampled_collision_increment": 3.0,
    "argmax_collision_increment": 5.0,
    "duplicate_collision_cap": 8,
    "initial_bc_learning_rate": 0.0003,
    "risk_bc_learning_rate": 0.0001,
    "initial_bc_batch_size": 256,
    "initial_tree_loss_weight": 0.05,
    "initial_gate_loss_weight": 0.2,
    "initial_bound_loss_weight": 0.0005,
    "initial_bc_gradient_clip": 0.5,
    "risk_bc_gradient_clip": 0.5,
    "collision_critic_version": "warehouse-v6-collision-critic.v1",
    "collision_critic_epochs": 80,
    "collision_critic_batch_size": 256,
    "collision_critic_learning_rate": 0.001,
    "collision_critic_positive_weight": 3.0,
    "collision_critic_seed_offset": 6_200,
    "collision_critic_environment_reset_count": 2,
    # A certificate-first fit to one frozen warmup NN initializes the
    # trainable Actor tree. It is not a program qualification set.
    "warmup_fit_sampled_episodes": 25,
    "warmup_fit_argmax_episodes": 25,
    "post_refit_bc_epochs": 8,
    "post_refit_warmup_masked_kl_weight": 0.0,
    # Only after all BC updates are complete, freeze the full Actor and
    # collect four disjoint internal development-purpose blocks.
    "final_fit_sampled_episodes": 25,
    "final_fit_argmax_episodes": 25,
    "final_calibration_sampled_episodes": 13,
    "final_calibration_argmax_episodes": 12,
    "final_eligibility_sampled_episodes": 13,
    "final_eligibility_argmax_episodes": 12,
    "final_validation_sampled_episodes": 12,
    "final_validation_argmax_episodes": 12,
    "warmup_fit_source": "frozen_warmup_full_nn_masked_distribution",
    "final_internal_source": "one_final_frozen_full_bc_actor",
    "program_fit_method": REGISTERED_PROGRAM_FIT_METHOD,
    "actor_model_version": "warehouse-dual-actor.v6.17-risk-protected",
    "core_bound_mode": "fit_protected_margin_cap_v3",
    "actor_topology_refresh": {
        "bc": "after_each_fit_stage",
        "certificate_invalidation": "each_refresh",
        "freeze_percentage": TOPOLOGY_FREEZE_PERCENTAGE,
        "ppo_percentages": list(TOPOLOGY_REFRESH_PERCENTAGES),
    },
    "feature_schema_sha256": "226d9b1374d8470c9051fcc319d0e19f8be94d7151bfc1a2b333905a5fd0d066",
    "action_physics_config_sha256": "ead2694c98fc8b9c7c6cb5dfdd1fbf9a6028a7138494c1bd38f3454b0d98fe67",
    "executable_recipe_file_sha256": "6998d0c68152a7161762b5f66e4e2d8b4136ea6bec400e8c9e830e18af8d6bec",
}
REGISTERED_RISK_DUAL_CONFIG = {
    "target": 0.04, "window_episodes": 128, "minimum_episodes": 64,
    "confidence": 0.90, "release_upper": 0.025,
    "ascent_gain": 2.0, "descent_gain": 1.0,
    "max_ascent_per_update": 0.10, "max_descent_per_update": 0.05,
    "max_multiplier": 20.0, "initial_multiplier": 1.0,
}


def _hash_json(value: object) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


REGISTERED_BC_SPEC_SHA256 = _hash_json(REGISTERED_BC_SPEC)
REGISTERED_RISK_DUAL_CONFIG_SHA256 = _hash_json(REGISTERED_RISK_DUAL_CONFIG)


def bc_data_seed_schedule(pair_seed: int) -> dict[str, list[int]]:
    """All 1,394 BC data episodes; critic resets are separately bound.

    The 50 warmup fit episodes train the Actor tree. The 124 final BC
    internal episodes have four distinct purposes and are never training rows.
    Their identities also differ from the 372 post-PPO program episodes.
    """
    if pair_seed not in (*EXPLORATORY_PAIR_SEEDS,
                         *DEVELOPMENT_PAIR_SEEDS, *FORMAL_PAIR_SEEDS):
        raise ValueError("unregistered Warehouse v6 pair seed")
    counts = {
        "initial_public_teacher": (10, REGISTERED_BC_SPEC["bc_episodes"]),
        **{f"initial_nn_dagger_{index}":
           (10 + index, REGISTERED_BC_SPEC["dagger_episodes"])
           for index in range(1, REGISTERED_BC_SPEC["dagger_rounds"] + 1)},
        "risk_public_teacher": (20, REGISTERED_BC_SPEC["risk_teacher_episodes"]),
        "risk_nn_visitation": (21, REGISTERED_BC_SPEC["risk_visit_episodes"]),
        "large_public_teacher": (30, REGISTERED_BC_SPEC["large_teacher_episodes"]),
        "large_nn_visitation": (31, REGISTERED_BC_SPEC["large_visit_episodes"]),
        "deep_public_teacher": (40, REGISTERED_BC_SPEC["deep_teacher_episodes"]),
    }
    for index in range(1, REGISTERED_BC_SPEC["deep_dagger_rounds"] + 1):
        for kind, offset in (("sampled", -1), ("argmax", 0)):
            counts[f"deep_{index}_{kind}"] = (
                40 + 2 * index + offset,
                REGISTERED_BC_SPEC["deep_dagger_episodes"] // 2)
    counts.update({
        "topology_fit_sampled":
            (60, REGISTERED_BC_SPEC["warmup_fit_sampled_episodes"]),
        "topology_fit_argmax":
            (61, REGISTERED_BC_SPEC["warmup_fit_argmax_episodes"]),
        "final_bc_precheck_fit_sampled":
            (70, REGISTERED_BC_SPEC["final_fit_sampled_episodes"]),
        "final_bc_precheck_fit_argmax":
            (71, REGISTERED_BC_SPEC["final_fit_argmax_episodes"]),
        "final_bc_precheck_calibration_sampled":
            (72, REGISTERED_BC_SPEC["final_calibration_sampled_episodes"]),
        "final_bc_precheck_calibration_argmax":
            (73, REGISTERED_BC_SPEC["final_calibration_argmax_episodes"]),
        "final_bc_precheck_eligibility_sampled":
            (74, REGISTERED_BC_SPEC["final_eligibility_sampled_episodes"]),
        "final_bc_precheck_eligibility_argmax":
            (75, REGISTERED_BC_SPEC["final_eligibility_argmax_episodes"]),
        "final_bc_precheck_validation_sampled":
            (76, REGISTERED_BC_SPEC["final_validation_sampled_episodes"]),
        "final_bc_precheck_validation_argmax":
            (77, REGISTERED_BC_SPEC["final_validation_argmax_episodes"]),
    })
    schedule = {name: [pair_seed * 100_000 + stage * 1_000 + index
                       for index in range(count)]
                for name, (stage, count) in counts.items()}
    flattened = [seed for values in schedule.values() for seed in values]
    if (len(counts) != 26 or len(flattened) != 1_394
            or len(set(flattened)) != len(flattened)):
        raise RuntimeError("Warehouse registered BC stage seed schedule drifted")
    return schedule


@lru_cache(maxsize=None)
def _reserved_critic_reset_seeds(release_id: str) -> frozenset[int]:
    sealed = sealed_seed_blocks(release_id)
    reserved = set((*EXPLORATORY_PAIR_SEEDS,
                    *DEVELOPMENT_PAIR_SEEDS, *FORMAL_PAIR_SEEDS,
                    *DEVELOPMENT_SELECTION_SEEDS,
                    *DEVELOPMENT_STRESS_BASE_SEEDS,
                    *DEVELOPMENT_PROGRAM_AUDIT_SEEDS,
                    *FORMAL_SELECTION_SEEDS,
                    *FORMAL_PROGRAM_PREAUDIT_SEEDS))
    reserved.update(seed for _, scenario, action in stress_schedule()
                    for seed in (scenario, action))
    for minor in range(int(release_id.rsplit(".v6.", 1)[1]) + 1):
        previous = sealed_seed_blocks(
            f"policylens-three-domain-20261016.v6.{minor}")
        reserved.update(previous["final_program_audit"])
        reserved.update(previous["heldout"])
        reserved.update(task_scenario_seed(seed, task)
                        for seed in previous["heldout"] for task in TASKS)
    for seed in (*EXPLORATORY_PAIR_SEEDS,
                 *DEVELOPMENT_PAIR_SEEDS, *FORMAL_PAIR_SEEDS):
        reserved.update(item for block in bc_data_seed_schedule(seed).values()
                        for item in block)
    for seed in (*DEVELOPMENT_PAIR_SEEDS, *FORMAL_PAIR_SEEDS):
        reserved.update(item for block in
                        final_program_fit_seed_schedule(release_id, seed).values()
                        for item in block)
        reserved.update(scenario_seed(
            "warehouse-v6-matched-120-step-blocks", seed, step)
            for step in range(0, max(BUDGETS), SCENARIO_BLOCK_STEPS))
    reserved.update(sealed["final_program_audit"])
    return frozenset(reserved)


def bc_critic_reset_schedule(pair_seed: int, release_id: str) -> dict[str, int]:
    """Two real env.reset calls used by the offline critic event enumerator."""
    if pair_seed not in (*EXPLORATORY_PAIR_SEEDS,
                         *DEVELOPMENT_PAIR_SEEDS, *FORMAL_PAIR_SEEDS):
        raise ValueError("unregistered Warehouse v6 pair seed")
    seed = pair_seed + REGISTERED_BC_SPEC["collision_critic_seed_offset"]
    if seed in _reserved_critic_reset_seeds(release_id):
        raise RuntimeError("Warehouse collision critic reset seed overlaps protocol")
    return {"seed": seed,
            "environment_reset_count": REGISTERED_BC_SPEC[
                "collision_critic_environment_reset_count"]}


def registered_method_lock_fields(program_fit_method: str) -> dict:
    """Method and paired BC seed commitments for a development budget lock."""
    if (_hash_json(REGISTERED_BC_SPEC) != REGISTERED_BC_SPEC_SHA256
            or _hash_json(REGISTERED_RISK_DUAL_CONFIG) !=
            REGISTERED_RISK_DUAL_CONFIG_SHA256):
        raise RuntimeError("Warehouse registered BC or risk recipe mutated")
    if program_fit_method not in PROGRAM_FIT_METHOD_CANDIDATES:
        raise ValueError("Warehouse candidate program method is unsupported")
    return {
        "program_fit_method": program_fit_method,
        "bc_recipe_sha256": REGISTERED_BC_SPEC_SHA256,
        "risk_dual_config_sha256": REGISTERED_RISK_DUAL_CONFIG_SHA256,
        "bc_data_seed_schedule_sha256": {
            str(seed): _hash_json(bc_data_seed_schedule(seed))
            for seed in (*DEVELOPMENT_PAIR_SEEDS, *FORMAL_PAIR_SEEDS)},
        "bc_critic_reset_schedule_sha256": {
            str(seed): _hash_json(bc_critic_reset_schedule(
                seed, INITIAL_RELEASE_ID))
            for seed in (*DEVELOPMENT_PAIR_SEEDS, *FORMAL_PAIR_SEEDS)},
    }


def bc_stage_receipt_shape() -> tuple[tuple[str, int | None], ...]:
    """Ordered v6.14 BC data, fit and internal precheck receipt stages."""
    stages: list[tuple[str, int | None]] = [
        ("initial_public_teacher", None),
        ("initial_public_bc", REGISTERED_BC_SPEC["bc_epochs"]),
    ]
    for index in range(1, REGISTERED_BC_SPEC["dagger_rounds"] + 1):
        stages += [
            (f"initial_nn_dagger_{index}", None),
            (f"initial_nn_dagger_{index}_fit",
             REGISTERED_BC_SPEC["dagger_epochs"]),
        ]
    stages += [
        ("topology_fit_sampled", None),
        ("topology_fit_argmax", None),
        ("post_refit_public_bc", REGISTERED_BC_SPEC["post_refit_bc_epochs"]),
    ]
    stages += [
        ("risk_public_teacher", None), ("risk_nn_visitation", None),
        ("risk_public_bc_fit", REGISTERED_BC_SPEC["risk_epochs"]),
        ("large_public_teacher", None), ("large_nn_visitation", None),
        ("large_public_bc_fit", REGISTERED_BC_SPEC["risk_epochs"]),
        ("deep_public_teacher", None),
    ]
    for index in range(1, REGISTERED_BC_SPEC["deep_dagger_rounds"] + 1):
        stages += [
            (f"deep_{index}_sampled", None),
            (f"deep_{index}_argmax", None),
            (f"deep_dagger_{index}_fit", REGISTERED_BC_SPEC["deep_epochs"]),
        ]
    stages.append(("final_standard_margin_calibration",
                   REGISTERED_BC_SPEC["deep_epochs"]))
    for purpose in ("fit", "calibration", "eligibility", "validation"):
        stages += [(f"final_bc_precheck_{purpose}_sampled", None),
                   (f"final_bc_precheck_{purpose}_argmax", None)]
    return tuple(stages)


def bc_topology_refresh_stages() -> tuple[str, ...]:
    """Ordered BC fit boundaries at which the current Actor is recertified."""
    return (
        "initial_public_bc",
        *(f"initial_nn_dagger_{index}_fit"
          for index in range(1, REGISTERED_BC_SPEC["dagger_rounds"] + 1)),
        "certificate_first_fit_only_tree_initialization",
        "post_refit_public_bc", "risk_public_bc_fit", "large_public_bc_fit",
        *(f"deep_dagger_{index}_fit"
          for index in range(1, REGISTERED_BC_SPEC["deep_dagger_rounds"] + 1)),
        "final_standard_margin_calibration",
    )


def bc_topology_refresh_history_valid(history: object) -> bool:
    """Receipt-level check for every BC-stage topology/certificate refresh."""
    if not isinstance(history, list):
        return False
    stages = bc_topology_refresh_stages()
    if len(history) != len(stages):
        return False
    for row, stage in zip(history, stages, strict=True):
        if (not isinstance(row, Mapping) or row.get("stage") != stage
                or row.get("route_parity_passed") is not True
                or row.get("path_legality_passed") is not True
                or row.get("core_qualification_recomputed") is not True):
            return False
        roles, hashes = row.get("roles"), row.get("role_tree_sha256")
        if (not isinstance(roles, Mapping) or set(roles) != set(ROLES)
                or not isinstance(hashes, Mapping)
                or set(hashes) != set(ROLES)):
            return False
        for role in ROLES:
            item = roles[role]
            if (not isinstance(item, Mapping)
                    or item.get("topology_sha256") != hashes[role]
                    or not isinstance(hashes[role], str)
                    or not re.fullmatch(r"[0-9a-f]{64}", hashes[role])
                    or type(item.get("training_scene_count")) is not int
                    or item["training_scene_count"] < 40
                    or not isinstance(item.get("qualification"), Mapping)
                    or item["qualification"].get("method") !=
                    REGISTERED_ACTOR_CORE_QUALIFICATION_METHOD):
                return False
    return True


def bc_report_valid(report: Mapping, pair_seed: int,
                    source_closure_sha256: str, *, legacy_v614: bool = False) -> bool:
    """Verify the actual 26-block BC receipt and independent internal audit.

    This is a pure receipt check. The release packager separately validates
    artifact bytes and the source closure; this function never opens a model.
    """
    if not isinstance(report, Mapping):
        return False
    try:
        if ((not legacy_v614 and
             _hash_json(REGISTERED_BC_SPEC) != REGISTERED_BC_SPEC_SHA256)
                or _hash_json(REGISTERED_RISK_DUAL_CONFIG) !=
                REGISTERED_RISK_DUAL_CONFIG_SHA256):
            return False
        schedule = bc_data_seed_schedule(pair_seed)
        critic_resets = bc_critic_reset_schedule(pair_seed, INITIAL_RELEASE_ID)
        recipe_version = (LEGACY_V614_BC_RECIPE_VERSION if legacy_v614
                          else REGISTERED_BC_RECIPE_VERSION)
        recipe_sha = (LEGACY_V614_BC_SPEC_SHA256 if legacy_v614
                      else REGISTERED_BC_SPEC_SHA256)
        recipe_file_sha = (LEGACY_V614_RECIPE_FILE_SHA256 if legacy_v614 else
                           REGISTERED_BC_SPEC["executable_recipe_file_sha256"])
        if (report.get("bc_recipe_version") != recipe_version
                or report.get("bc_recipe_sha256") != recipe_sha
                or report.get("executable_recipe_file_sha256") !=
                recipe_file_sha
                or report.get("risk_dual_config_sha256") !=
                REGISTERED_RISK_DUAL_CONFIG_SHA256
                or report.get("source_closure_sha256") != source_closure_sha256
                or report.get("feature_schema_sha256") !=
                REGISTERED_BC_SPEC["feature_schema_sha256"]
                or report.get("action_physics_config_sha256") !=
                REGISTERED_BC_SPEC["action_physics_config_sha256"]
                or report.get("bc_data_seed_blocks") != schedule
                or report.get("bc_data_seed_schedule_sha256") !=
                _hash_json(schedule)
                or report.get("bc_critic_reset_schedule") != critic_resets
                or report.get("bc_critic_reset_schedule_sha256") !=
                _hash_json(critic_resets)
                or report.get("teacher_uses_hidden_goal_or_route_state") is not False
                or report.get("offline_critic_used_at_execution") is not False
                or report.get(
                    "topology_calibration_eligibility_validation_used_for_bc_updates")
                is not False):
            return False
        critic = report.get("collision_critic")
        if (not isinstance(critic, Mapping)
                or critic.get("version") !=
                REGISTERED_BC_SPEC["collision_critic_version"]
                or critic.get("epochs") !=
                REGISTERED_BC_SPEC["collision_critic_epochs"]
                or critic.get("batch_size") !=
                REGISTERED_BC_SPEC["collision_critic_batch_size"]
                or critic.get("learning_rate") !=
                REGISTERED_BC_SPEC["collision_critic_learning_rate"]
                or critic.get("positive_weight") !=
                REGISTERED_BC_SPEC["collision_critic_positive_weight"]
                or critic.get("seed") != critic_resets["seed"]
                or critic.get("environment_reset_seed") !=
                critic_resets["seed"]
                or critic.get("environment_reset_count") !=
                critic_resets["environment_reset_count"]
                or critic.get("purpose") != "offline_BC_gradient_only_not_execution"
                or critic.get("labels_source") !=
                "env.step_robot_collision_events_delta"
                or critic.get("legal_actions_rewritten_by_environment") != 0
                or critic.get("validation_passed") is not True):
            return False
        stages = report.get("bc_stages")
        if not isinstance(stages, list):
            return False
        if not legacy_v614 and not bc_topology_refresh_history_valid(
                report.get("bc_topology_refresh_history")):
            return False
        expected = bc_stage_receipt_shape()
        if len(stages) != len(expected):
            return False
        for row, (name, epochs) in zip(stages, expected, strict=True):
            if not isinstance(row, Mapping) or row.get("stage") != name:
                return False
            if epochs is None:
                if (row.get("episode_seeds") != schedule[name]
                        or row.get("kind") != ("public_teacher" if "teacher" in name
                                              else "dual_nn_visitation")
                        or row.get("post_policy_interventions", 0) != 0
                        or ("argmax" in name and row.get(
                            "training_only_argmax_visits") is not True)
                        or ("sampled" in name and row.get(
                            "training_only_argmax_visits") is not False)):
                    return False
            elif name.startswith("initial_") or name == "post_refit_public_bc":
                if row.get("epochs") != epochs:
                    return False
                if (name == "post_refit_public_bc" and
                        row.get("warmup_masked_kl_weight") !=
                        REGISTERED_BC_SPEC["post_refit_warmup_masked_kl_weight"]):
                    return False
            else:
                fit = row.get("fit")
                initial_risk = name == "risk_public_bc_fit"
                if (not isinstance(fit, Mapping) or fit.get("epochs") != epochs
                        or fit.get("batch_size") !=
                        REGISTERED_BC_SPEC["batch_size"]
                        or fit.get("risk_weight") !=
                        REGISTERED_BC_SPEC["risk_weight"]
                        or fit.get("margin_gap") !=
                        REGISTERED_BC_SPEC["margin_gap"]
                        or fit.get("margin_weight") !=
                        (0.0 if initial_risk else
                         REGISTERED_BC_SPEC["margin_weight"])):
                    return False
        warmup = report.get("warmup_frozen_nn_actor")
        final = report.get("final_bc_internal_precheck")
        final_receipt = report.get("final_bc_internal_precheck_receipt")
        roles = report.get("pretrained_role_sha256")
        if (not isinstance(warmup, Mapping) or not isinstance(final, Mapping)
                or not isinstance(final_receipt, Mapping)
                or not isinstance(roles, Mapping) or set(roles) != set(ROLES)
                or not isinstance(warmup.get("role_sha256"), Mapping)
                or set(warmup["role_sha256"]) != set(ROLES)
                or final.get("purpose") !=
                "development_internal_precheck_after_complete_bc"
                or final.get("passed") is not True
                or not isinstance(final.get("final_bc_actor"), Mapping)
                or final["final_bc_actor"].get("role_sha256") != roles
                or (not legacy_v614 and
                    final["final_bc_actor"].get("model_version") !=
                    REGISTERED_BC_SPEC["actor_model_version"])
                or final.get("action_physics_config_sha256") !=
                REGISTERED_BC_SPEC["action_physics_config_sha256"]
                or final.get("warmup_actor_sha256") != warmup.get("sha256")
                or not isinstance(final_receipt.get("path"), str)
                or not Path(final_receipt["path"]).is_absolute()
                or not isinstance(final_receipt.get("sha256"), str)
                or final_receipt["sha256"] != sha256(
                    (json.dumps(final, sort_keys=True, allow_nan=False)
                     + "\n").encode("utf-8")).hexdigest()):
            return False
        warmup_schedule = {name.removeprefix("topology_"): schedule[name]
                           for name in schedule if name.startswith("topology_")}
        final_schedule = {name.removeprefix("final_bc_precheck_"): schedule[name]
                          for name in schedule if name.startswith("final_bc_precheck_")}
        if (final.get("warmup_topology_fit_seed_schedule_sha256") !=
                _hash_json(warmup_schedule)
                or final.get("final_bc_seed_schedule_sha256") !=
                _hash_json(final_schedule)):
            return False
        topology_visits = report.get("topology_episode_parts")
        final_visits = final.get("final_bc_visits")
        if (not isinstance(topology_visits, Mapping)
                or set(topology_visits) != set(warmup_schedule)
                or not isinstance(final_visits, Mapping)
                or set(final_visits) != set(final_schedule)):
            return False
        for name, seeds in warmup_schedule.items():
            if topology_visits[name].get("scenario_seeds") != seeds:
                return False
        for name, seeds in final_schedule.items():
            if final_visits[name].get("scenario_seeds") != seeds:
                return False
        if (not isinstance(final.get("roles"), Mapping)
                or set(final["roles"]) != set(ROLES)):
            return False
        if not legacy_v614 and (final.get("quality_screen_seeds") != report.get("quality_screen_seeds")
                or not quality_pass(final.get("quality_screen", {}).get("summary", {}), development=True)):
            return False
        for role in ROLES:
            actor_core = final["roles"][role].get("actor_core_qualification")
            installed_mass = final["roles"][role].get(
                "installed_core_validation_mass")
            actionable_mass = final["roles"][role].get(
                "installed_core_expected_actionable_mass")
            if (final["roles"][role].get("actor_role_sha256") != roles[role]
                    or final["roles"][role].get("refit_report", {}).get(
                        "validation_passed") is not True
                    or (not legacy_v614 and (
                        not isinstance(actor_core, Mapping)
                        or actor_core.get("method") !=
                        REGISTERED_ACTOR_CORE_QUALIFICATION_METHOD
                        or actor_core.get("validation_passed") is not True
                        or type(installed_mass) not in (int, float)
                        or not math.isfinite(installed_mass)
                        or not 0.35 <= installed_mass <= 1.0
                        or type(actionable_mass) not in (int, float)
                        or not math.isfinite(actionable_mass)
                        or not 0.30 <= actionable_mass <= installed_mass))
                    or report.get("topology_source_nn_role_sha256", {}).get(
                        "fit", {}).get(role) != warmup["role_sha256"][role]):
                return False
        return True
    except (KeyError, AttributeError, TypeError, ValueError):
        return False


def legacy_v614_bc_report_valid(report: Mapping, pair_seed: int,
                                source_closure_sha256: str) -> bool:
    """Read-only verifier for historical v6.14 receipts, never v6.16 admission."""
    return bc_report_valid(report, pair_seed, source_closure_sha256,
                           legacy_v614=True)


def scenario_seed(namespace: str, pair_seed: int, joint_step: int) -> int:
    """Assigned public scenario for a fixed joint-step block."""
    if joint_step < 0:
        raise ValueError("joint_step must be nonnegative")
    message = f"{namespace}|{pair_seed}|{joint_step // SCENARIO_BLOCK_STEPS}".encode()
    return int.from_bytes(sha256(message).digest()[:8], "big") & 0x7fff_ffff


def scenario_schedule_sha256(namespace: str, pair_seed: int, joint_steps: int) -> str:
    """Hash the exact assigned scenario for every joint training step."""
    if joint_steps < 0:
        raise ValueError("joint_steps must be nonnegative")
    digest = sha256()
    for step in range(joint_steps):
        digest.update(scenario_seed(namespace, pair_seed, step).to_bytes(4, "little"))
    return digest.hexdigest()


def _require_disjoint() -> None:
    blocks = {
        "development_selection": set(DEVELOPMENT_SELECTION_SEEDS),
        "stress": set(DEVELOPMENT_STRESS_BASE_SEEDS),
        "development_program_audit": set(DEVELOPMENT_PROGRAM_AUDIT_SEEDS),
        "selection": set(FORMAL_SELECTION_SEEDS),
        "preaudit": set(FORMAL_PROGRAM_PREAUDIT_SEEDS),
        "audit": set(FINAL_PROGRAM_AUDIT_SEEDS),
        "heldout": set(HELDOUT_BASE_SEEDS),
    }
    for first, a in blocks.items():
        for second, b in blocks.items():
            if first < second and a & b:
                raise RuntimeError(f"Warehouse v6 seed blocks overlap: {first}, {second}")
    if set(DEVELOPMENT_PAIR_SEEDS) & set(FORMAL_PAIR_SEEDS):
        raise RuntimeError("Warehouse v6 development and formal pair seeds overlap")
    if set(EXPLORATORY_PAIR_SEEDS) & set((*DEVELOPMENT_PAIR_SEEDS,
                                          *FORMAL_PAIR_SEEDS)):
        raise RuntimeError("Warehouse exploratory pair seed entered formal protocol")


_require_disjoint()


def task_scenario_seed(base_seed: int, task: int) -> int:
    if task not in TASKS:
        raise ValueError("Warehouse task must be 1, 2, or 3")
    return int(base_seed) * 10 + task


def stress_action_seed(base_seed: int, task: int, stream: int) -> int:
    """Independent, reproducible action RNG for the same public scenario."""
    if base_seed not in DEVELOPMENT_STRESS_BASE_SEEDS or stream not in DEVELOPMENT_ACTION_STREAMS:
        raise ValueError("unregistered Warehouse v6 stress scenario or stream")
    task_scenario_seed(base_seed, task)
    data = f"warehouse-v6-stress|{base_seed}|{task}|{stream}".encode()
    return int.from_bytes(sha256(data).digest()[:8], "big") & 0x7fff_ffff


def stress_schedule() -> tuple[tuple[int, int, int], ...]:
    """(task, environment seed, action RNG seed), 2304 disjoint trajectories."""
    return tuple((task, task_scenario_seed(seed, task),
                  stress_action_seed(seed, task, stream))
                 for task in TASKS for seed in DEVELOPMENT_STRESS_BASE_SEEDS
                 for stream in DEVELOPMENT_ACTION_STREAMS)


def sealed_seed_blocks(release_id: str = INITIAL_RELEASE_ID) -> dict[str, tuple[int, ...]]:
    """Reserve fresh final-audit and heldout blocks for every v6 candidate.

    The original v6.0 blocks are fixed. A failed candidate can only advance to
    a new release ID, whose once-only seed blocks cannot alias earlier ones.
    """
    match = _RELEASE_PATTERN.fullmatch(release_id)
    if match is None:
        raise ValueError("unregistered Warehouse v6 release ID; canonical minor required")
    offset = 100_000 * int(match.group(1))
    return {
        "final_program_audit": tuple(seed + offset for seed in FINAL_PROGRAM_AUDIT_SEEDS),
        "heldout": tuple(seed + offset for seed in HELDOUT_BASE_SEEDS),
    }


def final_audit_seeds(release_id: str = INITIAL_RELEASE_ID) -> tuple[int, ...]:
    return sealed_seed_blocks(release_id)["final_program_audit"]


def heldout_base_seeds(release_id: str = INITIAL_RELEASE_ID) -> tuple[int, ...]:
    return sealed_seed_blocks(release_id)["heldout"]


def heldout_schedule(release_id: str = INITIAL_RELEASE_ID) -> dict[int, tuple[int, ...]]:
    base_seeds = sealed_seed_blocks(release_id)["heldout"]
    return {task: tuple(task_scenario_seed(seed, task)
                        for seed in base_seeds) for task in TASKS}


def final_program_fit_seed_schedule(release_id: str, pair_seed: int
                                    ) -> dict[str, list[int]]:
    """Fresh four-part program data fixed by release and paired run identity.

    Candidate directory names cannot change the samples used by a formal
    Actor's post-selection program. This schedule is distinct from every
    quality, stress, program-audit and sealed test block.
    """
    if pair_seed not in (*DEVELOPMENT_PAIR_SEEDS, *FORMAL_PAIR_SEEDS):
        raise ValueError("unregistered Warehouse v6 pair seed")
    sealed = sealed_seed_blocks(release_id)
    minor = int(release_id.rsplit(".v6.", 1)[1])
    beginning = 50_000_000 + minor * 1_000_000 + pair_seed * 1_000
    schedule, cursor = {}, beginning
    for name, count in FINAL_PROGRAM_SPLIT_COUNTS.items():
        schedule[name] = list(range(cursor, cursor + count))
        cursor += count
    chosen = {seed for block in schedule.values() for seed in block}
    reserved = (
        set(DEVELOPMENT_SELECTION_SEEDS)
        | set(DEVELOPMENT_STRESS_BASE_SEEDS)
        | {scenario for _, scenario, _ in stress_schedule()}
        | set(DEVELOPMENT_PROGRAM_AUDIT_SEEDS)
        | set(FORMAL_SELECTION_SEEDS)
        | set(FORMAL_PROGRAM_PREAUDIT_SEEDS)
        | set(sealed["final_program_audit"])
        | set(sealed["heldout"])
        | {task_scenario_seed(seed, task)
           for seed in sealed["heldout"] for task in TASKS}
    )
    if chosen & reserved:
        raise RuntimeError("Warehouse final program seeds overlap protocol blocks")
    return schedule


def online_program_extraction_steps(joint_steps: int) -> tuple[int, ...]:
    """Lock complete training-block boundaries before a paired run starts."""
    if joint_steps not in BUDGETS:
        raise ValueError("Warehouse online program budget must be preregistered")
    numerators = tuple(joint_steps * percent
                       for percent in ONLINE_PROGRAM_EXTRACTION_PERCENTAGES)
    if any(numerator % 100 for numerator in numerators):
        raise RuntimeError("Warehouse online extraction percentage is not integral")
    steps = tuple(numerator // 100 for numerator in numerators)
    if (any(step % SCENARIO_BLOCK_STEPS for step in steps)
            or any(later - earlier <
                   SCENARIO_BLOCK_STEPS * sum(FINAL_PROGRAM_SPLIT_COUNTS.values())
                   for earlier, later in zip(steps, steps[1:]))
            or not all(0 < step < joint_steps for step in steps)):
        raise RuntimeError("Warehouse online program cohorts overlap or split a block")
    return steps


def topology_refresh_steps(joint_steps: int) -> tuple[int, ...]:
    """Training-only tree refreshes precede the first online extraction."""
    if joint_steps not in BUDGETS:
        raise ValueError("Warehouse topology refresh budget must be preregistered")
    numerators = tuple(joint_steps * percent
                       for percent in TOPOLOGY_REFRESH_PERCENTAGES)
    if any(numerator % 100 for numerator in numerators):
        raise RuntimeError("Warehouse topology refresh percentage is not integral")
    steps = tuple(numerator // 100 for numerator in numerators)
    frozen = topology_frozen_at_step(joint_steps)
    if (any(step % SCENARIO_BLOCK_STEPS for step in steps)
            or any(not 0 < step < frozen for step in steps)
            or tuple(sorted(set(steps))) != steps):
        raise RuntimeError("Warehouse topology refresh crosses a scenario block")
    return steps


def topology_frozen_at_step(joint_steps: int) -> int:
    """Tree structure is fixed before the first 75% program extraction."""
    if joint_steps not in BUDGETS:
        raise ValueError("Warehouse topology freeze budget must be preregistered")
    numerator = joint_steps * TOPOLOGY_FREEZE_PERCENTAGE
    if numerator % 100:
        raise RuntimeError("Warehouse topology freeze percentage is not integral")
    step = numerator // 100
    if (step % SCENARIO_BLOCK_STEPS
            or step != online_program_extraction_steps(joint_steps)[0]):
        raise RuntimeError("Warehouse topology freeze differs from first extraction")
    return step


def topology_refresh_history_valid(history: object, *,
                                   joint_steps: int) -> bool:
    """Verify both role trees were recertified after each PPO refresh."""
    if not isinstance(history, list):
        return False
    try:
        expected_steps = topology_refresh_steps(joint_steps)
        if len(history) != len(expected_steps):
            return False
        for row, step in zip(history, expected_steps, strict=True):
            if (not isinstance(row, Mapping)
                    or set(row) != {"joint_step", "role_tree_sha256",
                                       "route_parity_passed",
                                       "path_legality_passed",
                                       "core_qualification_recomputed"}
                    or row["joint_step"] != step
                    or type(row["joint_step"]) is not int
                    or any(row[key] is not True for key in (
                        "route_parity_passed", "path_legality_passed",
                        "core_qualification_recomputed"))):
                return False
            roles = row["role_tree_sha256"]
            if (not isinstance(roles, Mapping) or set(roles) != set(ROLES)
                    or any(not isinstance(roles[role], str)
                           or not re.fullmatch(r"[0-9a-f]{64}", roles[role])
                           for role in ROLES)):
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def frozen_topology_valid(report: Mapping) -> bool:
    """Verify both role trees and their qualified leaves remain frozen."""
    if not isinstance(report, Mapping):
        return False
    frozen = report.get("frozen_topology_role_sha256")
    final = report.get("final_topology_role_sha256")
    frozen_leaves = report.get("frozen_core_leaf_logits_sha256")
    final_leaves = report.get("final_core_leaf_logits_sha256")
    frozen_actor = report.get("frozen_actor_role_sha256")
    qualification = report.get("topology_freeze_qualification")
    if (any(not isinstance(value, Mapping) or set(value) != set(ROLES)
            for value in (frozen, final, frozen_leaves, final_leaves,
                          frozen_actor))
            or report.get("topology_updates_after_freeze") != 0
            or type(report.get("topology_updates_after_freeze")) is not int
            or not isinstance(qualification, Mapping)
            or qualification.get("stage") !=
               f"ppo_freeze_{report.get('topology_frozen_at_step')}"
            or qualification.get("role_tree_sha256") != frozen
            or qualification.get("route_parity_passed") is not True
            or qualification.get("path_legality_passed") is not True
            or qualification.get("core_qualification_recomputed") is not True
            or not isinstance(qualification.get("roles"), Mapping)
            or set(qualification["roles"]) != set(ROLES)):
        return False
    for role in ROLES:
        if (not all(isinstance(value[role], str) and
                    re.fullmatch(r"[0-9a-f]{64}", value[role])
                    for value in (frozen, frozen_leaves, frozen_actor))
                or final[role] != frozen[role]
                or final_leaves[role] != frozen_leaves[role]):
            return False
        role_receipt = qualification["roles"][role]
        if not isinstance(role_receipt, Mapping):
            return False
        core = role_receipt.get("qualification")
        installed = role_receipt.get("installed_core")
        parity = role_receipt.get("parity")
        if (role_receipt.get("topology_sha256") != frozen[role]
                or role_receipt.get("qualification_bound_to_installed_actor")
                   is not True
                or role_receipt.get("qualification_failed_core_reset") is not False
                or not isinstance(core, Mapping)
                or core.get("method") !=
                   REGISTERED_ACTOR_CORE_QUALIFICATION_METHOD
                or core.get("validation_passed") is not True
                or core.get("frozen_role_sha256") != frozen_actor[role]
                or core.get("actor_topology_sha256") != frozen[role]
                or not isinstance(installed, Mapping)
                or not isinstance(parity, Mapping)
                or type(parity.get("routing_mismatches")) is not int
                or parity["routing_mismatches"] != 0):
            return False
        selected = core.get("selected_core_leaf_ids")
        proofs = core.get("leaf_path_proof")
        if (not isinstance(selected, list) or not selected
                or any(type(leaf) is not int or not 0 <= leaf < CORE_MASK_LEAVES
                       for leaf in selected)
                or len(selected) != len(set(selected))
                or type(installed.get("core_leaf_count")) is not int
                or installed["core_leaf_count"] != len(selected)
                or not isinstance(proofs, Mapping)):
            return False
        # The qualification receipt hashes a NumPy bool mask (one byte per
        # leaf); the installed Actor hashes the same mask as float32 (four
        # bytes per leaf). Both must be reconstructed from the selected IDs.
        mask = bytearray(CORE_MASK_LEAVES)
        for leaf in selected:
            mask[leaf] = 1
        qualified_digest = sha256(mask).hexdigest()
        installed_digest = sha256(struct.pack(
            f"={CORE_MASK_LEAVES}f", *(float(value) for value in mask)
        )).hexdigest()
        if (core.get("core_mask_sha256") != qualified_digest
                or installed.get("core_mask_sha256") != installed_digest):
            return False
        if {key for key, proof in proofs.items()
                if isinstance(proof, Mapping) and proof.get("selected") is True} != \
                {str(leaf) for leaf in selected}:
            return False
        for leaf in selected:
            proof = proofs.get(str(leaf))
            if (not isinstance(proof, Mapping)
                    or proof.get("selected") is not True
                    or proof.get("candidate") is not True
                    or proof.get("raw_top") not in
                       ("WAIT", "UP", "DOWN", "LEFT", "RIGHT")
                    or not isinstance(proof.get("proven_actions"), list)
                    or proof["raw_top"] not in proof["proven_actions"]):
                return False
    return True


@dataclass(frozen=True)
class PairRunSpec:
    candidate_id: str
    pair_seed: int
    joint_steps: int
    output_dir: str
    program_fit_method: str
    release_id: str = INITIAL_RELEASE_ID
    scenario_namespace: str = "warehouse-v6-matched-120-step-blocks"
    checkpoint_seeds: tuple[int, ...] = ()
    online_program_extraction_steps: tuple[int, ...] = ()
    topology_refresh_steps: tuple[int, ...] = ()
    topology_frozen_at_step: int = 0
    architecture_mode: str = REGISTERED_ARCHITECTURE_MODE
    bc_recipe_version: str = REGISTERED_BC_RECIPE_VERSION
    bc_recipe_sha256: str = REGISTERED_BC_SPEC_SHA256
    risk_dual_config_sha256: str = REGISTERED_RISK_DUAL_CONFIG_SHA256
    bc_data_seed_schedule_sha256: str = ""
    bc_critic_reset_schedule_sha256: str = ""

    def __post_init__(self) -> None:
        if (_hash_json(REGISTERED_BC_SPEC) != REGISTERED_BC_SPEC_SHA256
                or _hash_json(REGISTERED_RISK_DUAL_CONFIG) !=
                REGISTERED_RISK_DUAL_CONFIG_SHA256):
            raise RuntimeError("Warehouse registered BC or risk recipe mutated")
        if not self.candidate_id or "/" in self.candidate_id or ".." in self.candidate_id:
            raise ValueError("candidate_id must be a simple, nonempty identifier")
        if self.pair_seed not in (*DEVELOPMENT_PAIR_SEEDS, *FORMAL_PAIR_SEEDS):
            raise ValueError("unregistered Warehouse v6 pair seed")
        if self.joint_steps not in BUDGETS:
            raise ValueError("Warehouse v6 training budget must be preregistered")
        if not self.output_dir:
            raise ValueError("output_dir required")
        required = (DEVELOPMENT_SELECTION_SEEDS if
                    self.pair_seed in DEVELOPMENT_PAIR_SEEDS else
                    FORMAL_SELECTION_SEEDS)
        if self.checkpoint_seeds != required:
            raise ValueError("Warehouse v6 checkpoint seeds do not match phase")
        if (self.architecture_mode != REGISTERED_ARCHITECTURE_MODE
                or self.program_fit_method not in PROGRAM_FIT_METHOD_CANDIDATES
                or self.bc_recipe_version != REGISTERED_BC_RECIPE_VERSION
                or self.bc_recipe_sha256 != REGISTERED_BC_SPEC_SHA256
                or self.risk_dual_config_sha256 !=
                REGISTERED_RISK_DUAL_CONFIG_SHA256):
            raise ValueError("Warehouse registered Actor, program, BC or risk recipe mismatch")
        bc_schedule_sha = _hash_json(bc_data_seed_schedule(self.pair_seed))
        if self.bc_data_seed_schedule_sha256:
            if self.bc_data_seed_schedule_sha256 != bc_schedule_sha:
                raise ValueError("Warehouse registered BC stage seed schedule mismatch")
        else:
            object.__setattr__(self, "bc_data_seed_schedule_sha256", bc_schedule_sha)
        critic_schedule_sha = _hash_json(bc_critic_reset_schedule(
            self.pair_seed, self.release_id))
        if self.bc_critic_reset_schedule_sha256:
            if self.bc_critic_reset_schedule_sha256 != critic_schedule_sha:
                raise ValueError("Warehouse collision critic reset schedule mismatch")
        else:
            object.__setattr__(self, "bc_critic_reset_schedule_sha256",
                               critic_schedule_sha)
        scheduled = online_program_extraction_steps(self.joint_steps)
        if self.online_program_extraction_steps:
            if self.online_program_extraction_steps != scheduled:
                raise ValueError("Warehouse online program extraction schedule mismatch")
        else:
            object.__setattr__(self, "online_program_extraction_steps", scheduled)
        refresh = topology_refresh_steps(self.joint_steps)
        if self.topology_refresh_steps:
            if self.topology_refresh_steps != refresh:
                raise ValueError("Warehouse topology refresh schedule mismatch")
        else:
            object.__setattr__(self, "topology_refresh_steps", refresh)
        frozen = topology_frozen_at_step(self.joint_steps)
        if self.topology_frozen_at_step:
            if self.topology_frozen_at_step != frozen:
                raise ValueError("Warehouse topology freeze step mismatch")
        else:
            object.__setattr__(self, "topology_frozen_at_step", frozen)
        sealed_seed_blocks(self.release_id)
        final_program_fit_seed_schedule(self.release_id, self.pair_seed)

    def to_mapping(self) -> dict:
        row = asdict(self)
        row["bc_data_seed_schedule"] = bc_data_seed_schedule(self.pair_seed)
        row["bc_critic_reset_schedule"] = bc_critic_reset_schedule(
            self.pair_seed, self.release_id)
        schedule = final_program_fit_seed_schedule(self.release_id, self.pair_seed)
        row["final_program_fit_seed_schedule"] = schedule
        row["final_program_fit_seed_schedule_sha256"] = _hash_json(schedule)
        return row

    def sha256(self) -> str:
        # A moved artifact directory does not change the scientific run.
        return _hash_json({key: value for key, value in self.to_mapping().items()
                           if key != "output_dir"})

    def training_config_sha256(self, source_closure_sha256: str) -> str:
        """Independently derive the registered BC/checkpoint config identity."""
        if (not isinstance(source_closure_sha256, str)
                or len(source_closure_sha256) != 64
                or any(char not in "0123456789abcdef"
                       for char in source_closure_sha256)):
            raise ValueError("invalid Warehouse training source closure")
        config = {**self.to_mapping(), **REGISTERED_BC_SPEC,
                  "source_closure_sha256": source_closure_sha256}
        return _hash_json({key: value for key, value in config.items()
                           if key != "output_dir"})


def quality_pass(summary: Mapping, *, development: bool) -> bool:
    """Development has safety margin; release quality limits are unchanged."""
    limits = (5.5, 6.0, 0.05) if development else (5.0, 8.0, 0.10)
    try:
        deliveries = float(summary["mean_deliveries"])
        collisions = float(summary["mean_collisions"])
        shutdown = float(summary["shutdown_episode_rate"])
    except (KeyError, TypeError, ValueError):
        return False
    return (deliveries >= limits[0] and 0 <= collisions <= limits[1]
            and 0 <= shutdown <= limits[2]
            and all(map(_finite, (deliveries, collisions, shutdown))))


def _finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def program_pass(metrics: Mapping, *, purpose: str, development: bool) -> bool:
    if purpose not in ("feedback", "explanation"):
        raise ValueError("unknown Warehouse program audit purpose")
    minimum_coverage = 0.30 if development else 0.20
    minimum_agreement = ((0.93 if purpose == "feedback" else 0.95)
                         if development else
                         (0.85 if purpose == "feedback" else 0.90))
    maximum_kl = 0.25 if development else 0.35
    try:
        coverage = float(metrics["coverage"])
        actionable_coverage = float(metrics["actionable_coverage"])
        agreement = float(metrics["agreement"])
        kl = float(metrics["kl"])
        rows = int(metrics["rows"])
        actionable_rows = int(metrics["actionable_rows"])
        illegal = int(metrics["illegal_count"])
    except (KeyError, TypeError, ValueError):
        return False
    return (rows > 0 and actionable_rows > 0 and
            actionable_rows <= rows and illegal == 0 and
            all(map(_finite, (coverage, actionable_coverage, agreement, kl))) and
            minimum_coverage <= coverage <= 1 and
            minimum_coverage <= actionable_coverage <= coverage and
            minimum_agreement <= agreement <= 1 and 0 <= kl <= maximum_kl)


def online_program_cohorts_valid(extractions: object, *,
                                 joint_steps: int,
                                 training_episodes: object) -> bool:
    """Recompute the four purpose cohorts instead of trusting a report flag."""
    if not isinstance(extractions, Mapping) or set(extractions) != set(ROLES):
        return False
    if not isinstance(training_episodes, list) or not training_episodes:
        return False
    episodes = {}
    for episode in training_episodes:
        if not isinstance(episode, Mapping):
            return False
        key = episode.get("episode_key")
        seed = episode.get("seed")
        start = episode.get("start_joint_step")
        length = episode.get("joint_steps")
        if (not isinstance(key, str) or key in episodes
                or type(seed) is not int or type(start) is not int
                or type(length) is not int or start < 0 or length <= 0
                or start + length > joint_steps):
            return False
        parts = key.split(":")
        if (len(parts) != 2 or parts[0] != str(seed)
                or not parts[1].isdigit() or parts[1] != str(int(parts[1]))):
            return False
        episodes[key] = (seed, start)
    steps = online_program_extraction_steps(joint_steps)
    all_scenes: set[int] = set()
    all_episode_ids: set[str] = set()
    for index, step in enumerate(steps):
        reference = None
        for role in ROLES:
            rows = extractions[role]
            if not isinstance(rows, list) or len(rows) != len(steps):
                return False
            row = rows[index]
            if not isinstance(row, Mapping) or (
                    row.get("joint_steps"), row.get("window_start_joint_step"),
                    row.get("window_end_joint_step")) != (
                    step, 0 if index == 0 else steps[index - 1], step):
                return False
            scenes = row.get("training_split_scenario_seeds")
            episode_ids = row.get("training_split_episode_ids")
            counts = row.get("training_split_episode_counts")
            if (not isinstance(scenes, Mapping)
                    or not isinstance(episode_ids, Mapping)
                    or not isinstance(counts, Mapping)
                    or any(set(part) != set(FINAL_PROGRAM_SPLIT_COUNTS)
                           for part in (scenes, episode_ids, counts))):
                return False
            local_scenes: set[int] = set()
            local_ids: set[str] = set()
            for purpose, required in FINAL_PROGRAM_SPLIT_COUNTS.items():
                seeds, ids = scenes[purpose], episode_ids[purpose]
                if (not isinstance(seeds, list) or len(seeds) != required
                        or any(type(seed) is not int for seed in seeds)
                        or not isinstance(ids, list) or not ids
                        or any(not isinstance(item, str) for item in ids)
                        or counts[purpose] != len(ids)):
                    return False
                seed_set, id_set = set(seeds), set(ids)
                if (len(seed_set) != required or len(id_set) != len(ids)
                        or seed_set & local_scenes or id_set & local_ids):
                    return False
                try:
                    id_scenes = {int(item.split(":", 1)[0]) for item in ids}
                except ValueError:
                    return False
                if id_scenes != seed_set:
                    return False
                if any(item not in episodes
                       or episodes[item][0] not in seed_set
                       or not (row["window_start_joint_step"] <=
                               episodes[item][1] < step)
                       for item in ids):
                    return False
                local_scenes.update(seed_set)
                local_ids.update(id_set)
            if reference is None:
                reference = (scenes, episode_ids)
                if local_scenes & all_scenes or local_ids & all_episode_ids:
                    return False
                all_scenes.update(local_scenes)
                all_episode_ids.update(local_ids)
            elif reference != (scenes, episode_ids):
                return False
    return True


def paired_training_pass(variants: Mapping, *, joint_steps: int) -> bool:
    """Check identical starts/schedules and actual gradients for both roles."""
    if joint_steps not in BUDGETS or set(variants) != set(VARIANTS):
        return False
    raw, regularized = (variants[name] for name in VARIANTS)
    if not isinstance(raw, Mapping) or not isinstance(regularized, Mapping):
        return False
    extraction_steps = list(online_program_extraction_steps(joint_steps))
    for report in (raw, regularized):
        seed = report.get("pair_seed")
        if (report.get("joint_steps") != joint_steps or
                seed not in (*DEVELOPMENT_PAIR_SEEDS, *FORMAL_PAIR_SEEDS) or
                report.get("actor_decisions") != joint_steps * len(ROLES) or
                report.get("action_physics_config_sha256") !=
                REGISTERED_BC_SPEC["action_physics_config_sha256"] or
                report.get("architecture_mode") != REGISTERED_ARCHITECTURE_MODE or
                report.get("program_fit_method") not in
                PROGRAM_FIT_METHOD_CANDIDATES or
                report.get("bc_recipe_version") != REGISTERED_BC_RECIPE_VERSION or
                report.get("bc_recipe_sha256") != REGISTERED_BC_SPEC_SHA256 or
                report.get("risk_dual_config_sha256") !=
                REGISTERED_RISK_DUAL_CONFIG_SHA256 or
                report.get("bc_data_seed_schedule_sha256") !=
                _hash_json(bc_data_seed_schedule(seed)) or
                report.get("online_program_extraction_steps") != extraction_steps or
                report.get("topology_refresh_steps") !=
                list(topology_refresh_steps(joint_steps)) or
                report.get("topology_frozen_at_step") !=
                topology_frozen_at_step(joint_steps) or
                not topology_refresh_history_valid(
                    report.get("topology_refresh_history"),
                    joint_steps=joint_steps) or
                not frozen_topology_valid(report) or
                report.get("online_program_split_block_cycle") !=
                FINAL_PROGRAM_SPLIT_COUNTS or
                report.get("online_program_cohorts_disjoint") is not True or
                report.get("post_policy_interventions") != 0 or
                report.get("physical_no_legal_action_rows") != 0):
            return False
        initial = report.get("initial_role_sha256")
        selected = report.get("selected_role_sha256")
        changes = report.get("role_weight_change_norm")
        if (not isinstance(initial, Mapping) or not isinstance(selected, Mapping)
                or not isinstance(changes, Mapping)
                or set(initial) != set(ROLES) or set(selected) != set(ROLES)
                or set(changes) != set(ROLES) or
                selected[ROLES[0]] == selected[ROLES[1]]):
            return False
        for role in ROLES:
            try:
                change = float(changes[role])
            except (TypeError, ValueError):
                return False
            if (not isinstance(initial[role], str) or
                    not isinstance(selected[role], str) or
                    initial[role] == selected[role] or
                    not _finite(change) or change <= 0):
                return False
    if (raw.get("initial_role_sha256") != regularized.get("initial_role_sha256")
            or raw.get("program_fit_method") !=
            regularized.get("program_fit_method")
            or not raw.get("scenario_schedule_sha256")
            or raw.get("scenario_schedule_sha256") !=
            regularized.get("scenario_schedule_sha256")):
        return False
    if any(raw["selected_role_sha256"][role] ==
           regularized["selected_role_sha256"][role] for role in ROLES):
        return False
    if not online_program_cohorts_valid(
            regularized.get("program_extractions"), joint_steps=joint_steps,
            training_episodes=regularized.get("training_episodes")):
        return False
    if any(extraction.get("program_fit_method") !=
           regularized["program_fit_method"]
           for role in ROLES
           for extraction in regularized["program_extractions"][role]):
        return False
    for report, should_be_positive in ((raw, False), (regularized, True)):
        rows = report.get("feedback_rows")
        gradients = report.get("feedback_gradient_norm")
        if (not isinstance(rows, Mapping) or not isinstance(gradients, Mapping)
                or set(rows) != set(ROLES) or set(gradients) != set(ROLES)):
            return False
        for role in ROLES:
            try:
                count, norm = int(rows[role]), float(gradients[role])
            except (TypeError, ValueError):
                return False
            if (not _finite(norm) or
                    (should_be_positive and (count <= 0 or norm <= 0)) or
                    (not should_be_positive and (count != 0 or norm != 0))):
                return False
    return True


def pair_quality_score(variants: Mapping) -> float:
    """Predeclared maximin selection uses quality alone, never program audit."""
    scores = []
    for variant in VARIANTS:
        summary = variants[variant]["quality"]
        if not quality_pass(summary, development=False):
            raise ValueError("quality selection requires both variants to pass")
        scores.append(float(summary["mean_deliveries"])
                      - 0.25 * float(summary["mean_collisions"])
                      - 5.0 * float(summary["shutdown_episode_rate"]))
    return min(scores)


def development_pair_pass(report: Mapping) -> bool:
    """Both variants and both roles need margin on independent stress scenes."""
    try:
        pair_seed = report["pair_seed"]
        budget = report["budget"]
        training = report["training"]
        evaluation = report["evaluation"]
        audits = report["program_audit"]
        regularized_hashes = training["regularized"]["selected_role_sha256"]
    except (KeyError, TypeError):
        return False
    if (pair_seed not in DEVELOPMENT_PAIR_SEEDS or budget not in BUDGETS or
            report.get("stress_episodes") != len(stress_schedule()) or
            report.get("stress_schedule_sha256") != _hash_json(stress_schedule()) or
            not paired_training_pass(training, joint_steps=budget) or
            set(evaluation) != set(VARIANTS) or set(audits) != set(ROLES)):
        return False
    if not all(quality_pass(evaluation[variant], development=True)
               for variant in VARIANTS):
        return False
    for role in ROLES:
        audit = audits[role]
        if (audit.get("actor_role_sha256") != regularized_hashes[role]
                or audit.get("sample_count", 0) < 1_000
                or not audit.get("program_sha256")):
            return False
        if not all(program_pass(audit.get(purpose, {}),
                                purpose=purpose, development=True)
                   for purpose in ("feedback", "explanation")):
            return False
    return True


def choose_development_budget(reports_by_budget: Mapping[int, Sequence[Mapping]]) -> int | None:
    """Lock the smallest passing budget before any formal paired run starts."""
    for budget in BUDGETS:
        reports = reports_by_budget.get(budget, ())
        if (len(reports) == len(DEVELOPMENT_PAIR_SEEDS)
                and {row.get("pair_seed") for row in reports} ==
                set(DEVELOPMENT_PAIR_SEEDS)
                and all(row.get("budget") == budget and development_pair_pass(row)
                        for row in reports)):
            return budget
    return None


def select_formal_pair(reports: Sequence[Mapping]) -> Mapping:
    """Only the fixed 24-seed quality rows select one of the three pairs."""
    if (len(reports) != len(FORMAL_PAIR_SEEDS) or
            {row.get("pair_seed") for row in reports} != set(FORMAL_PAIR_SEEDS)):
        raise ValueError("formal selection requires the three preregistered pairs")
    passing = []
    for row in reports:
        if row.get("selection_seeds") != list(FORMAL_SELECTION_SEEDS):
            raise ValueError("formal checkpoint selection seed mismatch")
        try:
            score = pair_quality_score(row["variants"])
        except (KeyError, TypeError, ValueError):
            continue
        passing.append((score, -int(row["pair_seed"]), row))
    if not passing:
        raise ValueError("no formal pair passed both Warehouse quality gates")
    return max(passing, key=lambda item: (item[0], item[1]))[2]


def claim_once(path: str | Path, *, phase: str, candidate_id: str,
               actor_hashes: Mapping, seed_rows: Sequence[int],
               release_id: str = INITIAL_RELEASE_ID) -> dict:
    """Durably claim final audit or heldout seeds before the first rollout.

    A started claim is never reusable after a crash or a failed audit.  A new
    candidate must use a new version and fresh sealed seed blocks.
    """
    if phase not in ("final_program_audit", "heldout"):
        raise ValueError("only sealed phases require a one-use claim")
    expected = sealed_seed_blocks(release_id)[phase]
    if tuple(seed_rows) != expected:
        raise ValueError("one-use claim has wrong seeds")
    if not candidate_id or not actor_hashes:
        raise ValueError("candidate and actor hashes required")
    receipt = {"phase": phase, "candidate_id": candidate_id,
               "release_id": release_id,
               "actor_hashes": actor_hashes, "seeds": list(seed_rows),
               "seed_sha256": _hash_json(list(seed_rows))}
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return receipt
