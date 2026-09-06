"""Frozen-policy behavior audits; final-test results never select a model."""
from __future__ import annotations
import argparse
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import random
import numpy as np

from env.cooperative_kitchen import CooperativeKitchen, program_decision
from backend.cooperative_kitchen.policy import ACTIONS, NumpyKitchenPolicy
from backend.cooperative_kitchen.splits import BASES, seeds, scenario_fingerprint


@lru_cache(maxsize=4)
def optimization_fingerprints(train_scenarios=512):
    return frozenset(scenario_fingerprint(CooperativeKitchen(seed=seed, scenario_id="generated").public_view())
                     for training_seed in range(3) for seed in seeds("train", train_scenarios, training_seed))


@lru_cache(maxsize=8)
def evaluation_exclusions(split, train_scenarios=512):
    excluded = set(optimization_fingerprints(train_scenarios))
    if split == "final_test":
        # Reserve validation configurations, including replacement and recovery
        # candidates, before opening any final-test outcomes.
        base = BASES["validation"]
        for seed in [*range(base, base + 200), *range(base + 50_000, base + 50_200)]:
            excluded.add(scenario_fingerprint(CooperativeKitchen(seed=seed, scenario_id="generated").public_view()))
    return frozenset(excluded)


def evaluation_scenes(split, episodes, train_scenarios=512):
    excluded = set(evaluation_exclusions(split, train_scenarios))
    selected, rejected = [], []
    for seed in range(BASES[split], BASES[split] + 50_000):
        fingerprint = scenario_fingerprint(CooperativeKitchen(seed=seed, scenario_id="generated").public_view())
        if fingerprint in excluded:
            rejected.append({"seed": seed, "fingerprint": fingerprint})
            continue
        excluded.add(fingerprint)
        selected.append({"seed": seed, "scenario_id": "generated", "fingerprint": fingerprint})
        if len(selected) == episodes:
            return selected, rejected
    raise RuntimeError("Held-out kitchen scenario namespace exhausted")


def rollout(policy, seed, profile="efficient", baseline=False, scenario_id="generated"):
    env = CooperativeKitchen(seed=int(seed), scenario_id=scenario_id)
    rng = random.Random(int(seed) + 17)
    initial = scenario_fingerprint(env.public_view())
    events, delivery_steps = Counter(), []
    space_released = None
    recovery_handoff = False
    began_holding_soup = env.public_view()["actors"][1]["holding"] == "soup"
    while True:
        observed = env.observations()
        if baseline:
            action = rng.choice(ACTIONS)
        else:
            action = policy.act({"ai": observed["ai"]})[0]["ai"]
        human = program_decision(env, "human", profile=profile, rng=rng)["action"]
        before = env.public_view()
        result = env.step({"human": human, "ai": action}, include_state=False)
        after = env.public_view()
        for event in result["events"]:
            events[event["type"]] += 1
            if event["type"] == "serve": delivery_steps.append(after["turn"])
            if began_holding_soup and event["type"] == "pickup" and event.get("actor") == "human" and space_released is None:
                space_released = after["turn"]
            if (began_holding_soup and event["type"] == "drop" and event.get("actor") == "ai"
                    and event.get("item") == "soup" and space_released is not None
                    and after["turn"] - space_released <= 40):
                recovery_handoff = True
        if result["done"]:
            return {"seed": int(seed), "scenario_id": scenario_id, "scenario_fingerprint": initial,
                    "profile": profile, "orders": after["orders"], "steps": after["turn"],
                    "score": 100 * after["orders"] - after["turn"], "success": after["orders"] >= 2,
                    "delivery_steps": delivery_steps, "events": dict(events),
                    "recovery_eligible": began_holding_soup and space_released is not None,
                    "space_released_step": space_released, "recovery_handoff": recovery_handoff}


def evaluate_policy(policy, split="validation", episodes=60, include_random=False):
    if split not in {"validation", "final_test"}:
        raise ValueError("Behavior evaluation requires a held-out split")
    train_scenarios = int(policy.metadata.get("config", {}).get("train_scenarios", 512))
    selected_scenes, rejected_scenes = evaluation_scenes(split, episodes, train_scenarios)
    scene_seeds = [row["seed"] for row in selected_scenes]
    profiles = {profile: [rollout(policy, seed, profile) for seed in scene_seeds]
                for profile in ("efficient", "perturbed")}
    recovery = [row for row in profiles["efficient"] if row["recovery_eligible"]]
    # Distinct generated congestion scenes supplement sparse natural coverage.
    # Renaming four fixed layouts under ten seeds is not independent evidence.
    if len(recovery) < 10:
        used = set(evaluation_exclusions(split, train_scenarios))
        used.update(row["scenario_fingerprint"] for row in profiles["efficient"])
        for candidate_seed in range(BASES[split] + 50_000, BASES[split] + 50_200):
            probe = CooperativeKitchen(seed=candidate_seed, scenario_id="generated")
            initial = probe.public_view()
            fingerprint = scenario_fingerprint(initial)
            if initial["actors"][1]["holding"] != "soup" or fingerprint in used:
                continue
            used.add(fingerprint)
            recovery.append(rollout(policy, candidate_seed, scenario_id="generated"))
            if len(recovery) >= 10:
                break
    summary = {key + "_completion_rate": float(np.mean([row["success"] for row in rows])) for key, rows in profiles.items()}
    summary.update({key + "_mean_score": float(np.mean([row["score"] for row in rows])) for key, rows in profiles.items()})
    eligible = [row for row in recovery if row["recovery_eligible"]]
    summary["recovery_rate"] = float(np.mean([row["recovery_handoff"] for row in eligible])) if eligible else 0.0
    gate = summary["efficient_completion_rate"] >= .8 and summary["perturbed_completion_rate"] >= .5 and summary["recovery_rate"] >= .9 and len(eligible) >= 10
    result = {"schema": "cooperative_kitchen_behavior_audit_v1", "split": split, "actor_sha256": policy.artifact_sha256,
              "checkpoint_sha256": policy.checkpoint_id, "episodes_per_profile": episodes,
              "thresholds": {"efficient_completion_rate": .8, "perturbed_completion_rate": .5, "recovery_rate": .9},
              "summary": summary, "training_gate": bool(gate), "profiles": profiles, "recovery": recovery,
              "scenario_selection": {"selected": selected_scenes, "rejected_duplicate_initial_states": rejected_scenes,
                                     "excludes_all_three_optimization_seed_pools": True},
              "recovery_definition": "AI returns the initially held soup within 40 joint steps after the human frees a shared counter; no runtime override"}
    if include_random:
        random_rows = [rollout(policy, seed, baseline=True) for seed in scene_seeds]
        result["random_baseline"] = {"completion_rate": float(np.mean([row["success"] for row in random_rows])), "episodes": random_rows}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actor"); parser.add_argument("--split", choices=("validation", "final_test"), default="validation")
    parser.add_argument("--episodes", type=int, default=60); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = evaluate_policy(NumpyKitchenPolicy(args.actor), args.split, args.episodes, True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
