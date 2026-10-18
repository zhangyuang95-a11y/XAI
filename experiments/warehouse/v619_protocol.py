"""Nonpromotable fixed-Actor development protocol; no training or sealed claims.

New scene reservations require a completed read-only ledger before execution.
The old v6.18 training protocol and its Actor initialization are not changed.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

CANDIDATE_ID = "warehouse-v619-01"
METHOD = "path_constrained_distribution_tree_v1"
ACTOR_SHA256 = "76acb00d8d24c5540379513a89f7c1024f5849427858fe945a9c39cd67a3fd88"
BASELINE_RUN = "output/warehouse_v618_preflight/warehouse-v618-quality-selective-01-bc-preflight/pair_6150_steps_200000"
OUTPUT_ROOT = "output/warehouse_v619"
FRESH_BASE = 100_266_000
INDEPENDENT_BASES = tuple(range(100_266_200, 100_266_224))
TASKS = (1, 2, 3)
ROLES = ("robot_1", "robot_2")
SPLITS = ("fit", "calibration", "eligibility", "validation")


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def scene_seed(base, task):
    if type(base) is not int or base < 0 or type(task) is not int or task not in TASKS:
        raise ValueError("invalid v6.19 scene identity")
    return base*10+task


def independent_action_seed(base, task, stream):
    if (type(base) is not int or base not in INDEPENDENT_BASES
            or type(task) is not int or task not in TASKS
            or type(stream) is not int or stream not in (0, 1, 2)):
        raise ValueError("unregistered v6.19 independent draw")
    # A reserved contiguous block is auditable against historical seed ledgers.
    return 1_700_000_000 + (base-INDEPENDENT_BASES[0])*9 + (task-1)*3 + stream


def seed_plan():
    parts = {}
    for name, offset, count in (("fit", 0, 50), ("calibration", 50, 25),
                               ("eligibility", 75, 25), ("validation", 100, 24)):
        parts[name] = [{"base": FRESH_BASE+offset+i,
            "task": (offset+i) % 3+1,
            "scenario_seed": scene_seed(FRESH_BASE+offset+i, (offset+i) % 3+1),
            "sampling_mode": "argmax" if name == "fit" and i >= 25 else "sampled"}
            for i in range(count)]
    independent = [{"base": base, "task": task, "stream": stream,
                    "scenario_seed": scene_seed(base, task),
                    "action_seed": independent_action_seed(base, task, stream),
                    "sampling_mode": "sampled"}
                   for base in INDEPENDENT_BASES for task in TASKS for stream in (0, 1, 2)]
    scenarios = [item["scenario_seed"] for values in parts.values() for item in values]
    if (len(set(scenarios)) != 124
            or set(scenarios) & {r["scenario_seed"] for r in independent}
            or len({r["action_seed"] for r in independent}) != 216):
        raise RuntimeError("v6.19 purpose/draw seed collision")
    return {"splits": parts, "independent": independent,
            "environment_seed_rule": "base_times_10_plus_task",
            "fit_sampling": "25_sampled_plus_25_argmax",
            "nonfit_sampling": "natural_nn_sampled_all_physical_frames"}


def executable_recipe():
    from backend.training.warehouse_v619_tree import PRODUCTION_SEARCH_CONFIG
    return {"version": "warehouse-v619-program-recipe.v1", "candidate_id": CANDIDATE_ID,
            "method": METHOD, "fixed_actor_sha256": ACTOR_SHA256,
            "primary_comparison_arm": "P2", "all_arms": ["P0", "P1", "P2"],
            "leaf_fit": {"epsilon": .001, "regularization": .0001,
                         "maxiter": 200, "tolerance": .000001},
            "calibration_temperatures": [.8, .9, 1., 1.1, 1.2],
            "search": asdict(PRODUCTION_SEARCH_CONFIG),
            "seed_plan": seed_plan(),
            "bc_gates": {"leaf_coverage": .35, "expected_feedback_coverage": .30,
                "feedback_agreement": .93, "explanation_agreement": .95,
                "maximum_kl": .25, "whole_tree_raw_illegal_count": 0,
                "qualified_path_uncertified_count": 0},
            "independent_gates": {"leaf_coverage": .30, "actual_actionable_coverage": .30,
                "feedback_agreement": .93, "explanation_agreement": .95,
                "maximum_kl": .25, "whole_tree_raw_illegal_count": 0,
                "deliveries": 5.5, "collisions": 6., "shutdown_rate": .05},
            "publishable": False, "PPO_joint_steps": 0,
            "actor_initialization_extractor_changed": False,
            "failed_validation_may_select_another_arm": False}


def validate_recipe(root: Path):
    path = root / "configs/warehouse_v619_program_recipe.json"
    raw = path.read_bytes()
    if json.loads(raw) != json.loads(json.dumps(executable_recipe())):
        raise RuntimeError("v6.19 executable program recipe drifted")
    return sha256(raw).hexdigest()
