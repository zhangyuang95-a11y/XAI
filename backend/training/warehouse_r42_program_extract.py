"""Extract and audit a compact post-hoc r4.2 decision-tree program."""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import random

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from backend.training.warehouse_r41_diagnostic_conflict_play_selection import (
    actor_environment,
)
from backend.warehouse_r42_program import VERSION, digest
from env.warehouse.navigation import ACTIONS, shortest_path_distance
from env.warehouse_native.partners import partner_action
from env.warehouse_native.policy import NumPyNativeActor


EXTRACT_VERSION = "warehouse-r42-program-extraction.v1"
ACTION_INDEX = {action: index for index, action in enumerate(ACTIONS)}
PROFILES = ("wait", "skilled", "assertive", "noisy")


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _critical(env) -> bool:
    actor = env.state.by_id("robot_2")
    other = env.state.by_id("robot_1")
    distance = shortest_path_distance(
        actor.position, other.position, env.config.map_layout_id
    )
    return bool(actor.carrying_task_id or actor.battery <= 30 or distance <= 2)


def collect(actor, scenes, split: str):
    observations, labels, critical, branches = [], [], [], []
    for scene_index, scene in enumerate(scenes["splits"][split]):
        for profile_index, profile in enumerate(PROFILES):
            env = actor_environment(actor, scene)
            rng = random.Random(9_100_000 + scene_index * 31 + profile_index)
            for step in range(env.config.horizon):
                obs = env.observations()["robot_2"].astype(np.float32, copy=True)
                selected, _ = actor.act(env.observations(), deterministic=True)
                action = selected["robot_2"]
                observations.append(obs)
                labels.append(ACTION_INDEX[action])
                critical.append(_critical(env))
                if split == "conflict_validation" and step % 12 == 0:
                    snapshot = deepcopy(env.snapshot())
                    baseline = ACTION_INDEX[action]
                    for player_action in ACTIONS:
                        branch = actor_environment(actor, scene)
                        branch.restore(deepcopy(snapshot), require_feedback=True)
                        branch.step(
                            {"robot_1": player_action, "robot_2": action},
                            decision_metadata={"policy_action": action,
                                               "submitted_action": action,
                                               "post_policy_overrides": 0},
                        )
                        if branch.done:
                            continue
                        branch_obs = branch.observations()["robot_2"].astype(
                            np.float32, copy=True
                        )
                        branch_action, _ = actor.act(
                            branch.observations(), deterministic=True
                        )
                        branches.append((branch_obs,
                                         ACTION_INDEX[branch_action["robot_2"]],
                                         baseline))
                player_action = (
                    "WAIT" if profile == "wait" else
                    partner_action(env, "robot_1", profile, rng)
                )
                env.step(
                    {"robot_1": player_action, "robot_2": action},
                    decision_metadata={"policy_action": action,
                                       "submitted_action": action,
                                       "post_policy_overrides": 0},
                )
                if env.done:
                    break
    return (np.stack(observations), np.asarray(labels, dtype=np.int64),
            np.asarray(critical, dtype=bool), branches)


def _tree_payload(model):
    tree = model.tree_
    values = tree.value[:, 0, :]
    if values.shape[1] != len(ACTIONS):
        expanded = np.zeros((len(values), len(ACTIONS)), dtype=np.float64)
        for index, class_id in enumerate(model.classes_):
            expanded[:, int(class_id)] = values[:, index]
        values = expanded
    return {
        "children_left": tree.children_left.astype(int).tolist(),
        "children_right": tree.children_right.astype(int).tolist(),
        "feature": tree.feature.astype(int).tolist(),
        "threshold": tree.threshold.astype(float).tolist(),
        "value": values.astype(float).tolist(),
        "node_count": int(tree.node_count),
        "max_depth": int(tree.max_depth),
    }


def build(actor_path: Path, manifest_path: Path, output: Path):
    actor = NumPyNativeActor(actor_path)
    scenes = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_x, train_y, train_critical, _ = collect(actor, scenes, "train")
    valid_x, valid_y, valid_critical, branches = collect(
        actor, scenes, "conflict_validation"
    )
    candidates = []
    selected = None
    for depth, leaves in ((10, 256), (12, 512), (14, 1024), (16, 2048),
                          (20, 4096)):
        model = DecisionTreeClassifier(
            random_state=260915, max_depth=depth, max_leaf_nodes=leaves,
            min_samples_leaf=2, class_weight=None,
        )
        weights = np.where(train_critical, 2.5, 1.0)
        model.fit(train_x, train_y, sample_weight=weights)
        prediction = model.predict(valid_x)
        overall = float((prediction == valid_y).mean())
        nonwait = valid_y != ACTION_INDEX["WAIT"]
        non_wait_score = float((prediction[nonwait] == valid_y[nonwait]).mean())
        critical_score = float((prediction[valid_critical] == valid_y[valid_critical]).mean())
        if branches:
            branch_x = np.stack([row[0] for row in branches])
            branch_y = np.asarray([row[1] for row in branches])
            branch_base = np.asarray([row[2] for row in branches])
            branch_prediction = model.predict(branch_x)
            changed = branch_y != branch_base
            direction = float(((branch_prediction != branch_base)[changed]
                               == (branch_y != branch_base)[changed]).mean()) \
                if changed.any() else 1.0
        else:
            direction = 0.0
        result = {
            "max_depth_limit": depth, "max_leaf_nodes": leaves,
            "node_count": int(model.tree_.node_count),
            "overall_fidelity": overall,
            "non_wait_fidelity": non_wait_score,
            "critical_fidelity": critical_score,
            "effective_intervention_direction_accuracy": direction,
        }
        candidates.append(result)
        if (overall >= .90 and non_wait_score >= .90
                and critical_score >= .85 and direction >= .85):
            selected = (model, result)
            break
    if selected is None:
        model, result = model, candidates[-1]
    else:
        model, result = selected
    audit = {
        **result,
        "training_samples": int(len(train_y)),
        "validation_samples": int(len(valid_y)),
        "non_wait_validation_samples": int((valid_y != ACTION_INDEX["WAIT"]).sum()),
        "critical_validation_samples": int(valid_critical.sum()),
        "effective_intervention_samples": int(sum(
            row[1] != row[2] for row in branches
        )),
        "scenario_split_isolated": True,
        "program_controls_runtime_actions": False,
        "candidates": candidates,
    }
    payload = {
        "version": VERSION,
        "extraction_version": EXTRACT_VERSION,
        "actor_sha256": file_hash(actor_path),
        "actor_parameters_sha256": actor.metadata.get("actor_parameters_sha256"),
        "manifest_sha256": file_hash(manifest_path),
        "actions": list(ACTIONS),
        "feature_names": list(actor.metadata["feature_names"]),
        "tree": _tree_payload(model),
        "audit": audit,
    }
    payload["content_sha256"] = digest(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")) + "\n", encoding="utf-8")
    report = {
        "version": EXTRACT_VERSION,
        "passed": all((audit["overall_fidelity"] >= .90,
                       audit["non_wait_fidelity"] >= .90,
                       audit["critical_fidelity"] >= .85,
                       audit["effective_intervention_direction_accuracy"] >= .85)),
        "actor_sha256": payload["actor_sha256"],
        "program_sha256": file_hash(output),
        "program_content_sha256": payload["content_sha256"],
        "audit": audit,
    }
    report_path = output.with_name("program_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True,
                                      indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = build(Path(args.actor).resolve(), Path(args.manifest).resolve(),
                   Path(args.output).resolve())
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
