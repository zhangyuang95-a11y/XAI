#!/usr/bin/env python3
"""Read a retired candidate and diagnose the new topology on its spent scenes.

This is a design diagnostic, never admission evidence. No audit or holdout
seed can be supplied: all scenes must come from the retired BC receipt.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.training.warehouse_v6_actor import FrozenWarehouseV6Actor, WarehouseV6Actor
from backend.training.warehouse_v6_core_tree import _masked_probabilities
from backend.training.warehouse_v617_risk_tree import fit_risk_tree
from env.warehouse.domain import participant_study_config
from env.warehouse.environment import WarehouseMultiAgentEnv
from env.warehouse.navigation import ACTIONS
from env.warehouse.pure_nn import AGENTS, action_masks
from env.warehouse.public_observation_v6 import encode_public_observation
from env.warehouse.public_action_features_v617 import encode_actor_tree_features


def collect(actor, seeds):
    rows, outcomes = [], []
    for seed in seeds:
        env = WarehouseMultiAgentEnv(participant_study_config())
        env.reset(seed=seed)
        state = env.get_state()
        state.participant_controlled_agent_id = "robot_1"
        env.set_state(state)
        while not (env.state.terminated or env.state.truncated):
            obs, masks = encode_public_observation(env.state), action_masks(env.state)
            actions, _ = actor.act(env.state, base_seed=seed)
            rows.append((obs, masks, seed))
            _, _, _, _, info = env.step(actions)
            if info["proposed_actions"] != actions or info["action_override_agents"]:
                raise RuntimeError("capacity diagnostic NN action was rewritten")
        outcomes.append({"seed": seed, "deliveries": env.state.total_deliveries,
                         "collisions": env.state.robot_collision_events,
                         "shutdown_episode": bool(env.state.shutdown_count)})
    return (np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows]),
            np.asarray([r[2] for r in rows]), outcomes)


def summary(outcomes):
    return {"episodes": len(outcomes), **{name: float(np.mean([r[key] for r in outcomes]))
        for name, key in (("mean_deliveries", "deliveries"),
                          ("mean_collisions", "collisions"),
                          ("shutdown_episode_rate", "shutdown_episode"))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retired-precheck", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    receipt_path = Path(args.retired_precheck).resolve(strict=True)
    receipt = json.loads(receipt_path.read_bytes())
    if receipt.get("passed") is not False:
        raise ValueError("diagnostic requires a retired failed BC receipt")
    artifact = receipt["final_bc_actor"]
    old = FrozenWarehouseV6Actor.load(artifact["path"], expected_sha256=artifact["sha256"])
    torch.manual_seed(617)
    torch.set_num_threads(2)
    probe = WarehouseV6Actor()
    parts = {}
    for purpose in ("fit", "calibration", "eligibility", "validation"):
        seeds = [seed for kind in ("sampled", "argmax")
                 for seed in receipt["final_bc_visits"][f"{purpose}_{kind}"]["scenario_seeds"]]
        parts[purpose] = collect(old, seeds)
        print(json.dumps({"phase": purpose, "rows": len(parts[purpose][0]),
                          "quality": summary(parts[purpose][3])}), flush=True)
    role_reports = {}
    for index, role in enumerate(AGENTS):
        actor = probe.actors[role]
        # A counterfactual initialization only; never a registered BC start.
        old_weights = old.roles[role].weights
        state = actor.state_dict()
        for name, value in old_weights.items():
            if name.startswith("tree.") or name == "core_leaf_mask":
                continue
            if name in ("gate.0.weight", "action_residual.0.weight"):
                value = np.concatenate((value[:, :171], np.zeros((len(value), 2), np.float32),
                                         value[:, 171:]), axis=1)
            if name in state and state[name].shape == value.shape:
                state[name] = torch.from_numpy(value.copy())
        actor.load_state_dict(state)
        obs, masks, scenes, _ = parts["fit"]
        x = encode_actor_tree_features(obs[:, index], masks[:, index])
        p = _masked_probabilities(old.logits(obs, masks)[:, index], masks[:, index])
        fitted = fit_risk_tree(x, masks[:, index], p,
                              tuple(f"nn-visit:{seed}" for seed in scenes))
        actor.initialize_tree_from_core_result(fitted)
        frozen = FrozenWarehouseV6Actor.from_model(probe).roles[role]
        metrics = {}
        for purpose, (obs, masks, scenes, _) in parts.items():
            x = encode_actor_tree_features(obs[:, index], masks[:, index])
            m = masks[:, index]
            leaves = frozen.leaf(x)
            selected = fitted.core_mask[leaves]
            tree_p = _masked_probabilities(fitted.leaf_logits[leaves], m)
            old_p = _masked_probabilities(old.logits(obs, masks)[:, index], m)
            new_p = _masked_probabilities(frozen.logits_encoded(x), m)
            chosen = tree_p.argmax(1)
            per_target = {}
            for label, target in (("retired_nn", old_p), ("protected_actor_probe", new_p)):
                kl = np.sum(target * (np.log(np.maximum(target, 1e-9))-
                                      np.log(np.maximum(tree_p, 1e-9))), axis=1)
                per_target[label] = {
                    "protected_coverage": float(selected.mean()),
                    "agreement": float(np.mean(chosen[selected] == target[selected].argmax(1))) if selected.any() else None,
                    "masked_kl": float(kl[selected].mean()) if selected.any() else None,
                    "expected_actionable": float(np.mean(selected * (chosen == target.argmax(1)) *
                        target[np.arange(len(target)), chosen])),
                }
            metrics[purpose] = per_target
        role_reports[role] = {"fit_tree": fitted.report, "metrics": metrics}
    frozen_probe = FrozenWarehouseV6Actor.from_model(probe)
    seeds = [seed for kind in ("sampled", "argmax")
             for seed in receipt["final_bc_visits"][f"validation_{kind}"]["scenario_seeds"]]
    _, _, _, probe_outcomes = collect(frozen_probe, seeds)
    report = {"purpose": "retired_scenes_design_diagnostic_not_admission",
        "retired_precheck_sha256": sha256(receipt_path.read_bytes()).hexdigest(),
        "retired_actor_sha256": artifact["sha256"], "roles": role_reports,
        "same_scene_quality": {"retired_actor": summary(parts["validation"][3]),
                               "untrained_protected_probe": summary(probe_outcomes)},
        "probe_outcomes": probe_outcomes, "sealed_seeds_used": False,
        "joint_training_effect_measured": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(output), "quality": report["same_scene_quality"]}), flush=True)


if __name__ == "__main__":
    main()
