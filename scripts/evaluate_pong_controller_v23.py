#!/usr/bin/env python3
"""Frozen Actor comparison of v2.3 coordination and existing controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from domains.pong.training.evaluation import _episode, _initial_content_signature, _summarize
from domains.pong.environment.engine import PongEnvironment
from domains.pong.training.runner import PongPPOTrainer, _device, load_config


def _actor_hash(trainer: PongPPOTrainer) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(trainer.model.actor.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().astype("float32").tobytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/pong_controller_v23.json"))
    parser.add_argument("--output", default=str(ROOT / "output/pong_controller/v23"))
    parser.add_argument("--split", choices=("validation", "final_test"), default="validation")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    spec = json.loads(Path(args.config).read_text(encoding="utf-8"))
    source = ROOT / spec["source_run"]
    trainer = PongPPOTrainer(load_config(source / "config.yaml"), device=_device(args.device))
    trainer.load_state_dict(torch.load(source / spec["source_checkpoint"], map_location="cpu", weights_only=False))
    actor_hash = _actor_hash(trainer)
    if actor_hash != spec["actor_sha256"]:
        raise ValueError("frozen Actor hash differs from v2.3 comparison manifest")
    seeds = list(map(int, spec["validation_seeds" if args.split == "validation" else "final_test_seeds"]))
    if len(seeds) != len(set(seeds)) or set(seeds) & set(spec["final_test_seeds" if args.split == "validation" else "validation_seeds"]):
        raise ValueError("validation/final-test seed overlap")
    signatures = [_initial_content_signature(PongEnvironment(trainer.env_config, seed=s)) for s in seeds]
    opposite_seeds = spec["final_test_seeds" if args.split == "validation" else "validation_seeds"]
    opposite_signatures = {_initial_content_signature(PongEnvironment(trainer.env_config, seed=int(s)))
                           for s in opposite_seeds}
    if (len(signatures) != len(set(signatures)) or set(signatures) & opposite_signatures
            or set(signatures) & set(getattr(trainer, "training_initial_signatures", set()))):
        raise ValueError("comparison scenes repeat training or each other")
    plan = {
        "coordinated_nn_nn": ("coordinated", "nn_nn", None),
        "coordinated_rule_player": ("coordinated", "nn_rule", 0),
        "coordinated_perturb_player": ("coordinated", "nn_perturb", 0),
        "limited_nn_nn": ("hybrid", "nn_nn", None),
        "limited_rule_player": ("hybrid", "nn_rule", 0),
        "limited_perturb_player": ("hybrid", "nn_perturb", 0),
        "pure_nn_nn": ("pure_nn", "nn_nn", None),
        "pure_rule_player": ("pure_nn", "nn_rule", 0),
        "pure_perturb_player": ("pure_nn", "nn_perturb", 0),
        "rule_rule": ("rule_only", "rule_rule", None),
        "rule_perturb_player": ("rule_only", "rule_perturb", 0),
    }
    compact: dict[str, dict[str, object]] = {}
    per_seed: dict[str, list[dict[str, object]]] = {}
    for name, (mode, condition, external) in plan.items():
        episodes = []
        for seed in seeds:
            episodes.append(_episode(trainer, condition=condition, seed=seed, external_agent=external,
                                     controller_mode=mode,
                                     controller_settings=spec["controller"] if mode == "coordinated" else None))
        summary = _summarize(name, episodes)
        compact[name] = {key: summary[key] for key in (
            "mean_weighted_misses", "aggregate_small_success_rate", "aggregate_large_success_rate",
            "intervention_rate", "maximum_intervention_streak", "action_chain_valid")}
        compact[name]["left_large_after_ready"] = summary["diagnostics"].get("left_large_after_ready", 0)
        gaps = [gap for item in episodes for gap in item["large_ready_departures"]]
        compact[name]["ready_coverage_gaps"] = len(gaps)
        compact[name]["ready_gap_followed_by_miss"] = sum(gap["outcome"] == "missed" for gap in gaps)
        compact[name]["boundary_actions"] = summary["diagnostics"].get("boundary_actions", 0)
        timings = [ms for item in episodes for ms in item.get("coordinated_decision_ms", [])]
        compact[name]["planning_mean_ms"] = statistics.mean(timings) if timings else None
        compact[name]["planning_p95_ms"] = sorted(timings)[min(len(timings) - 1, int(.95 * len(timings)))] if timings else None
        per_seed[name] = [{"seed": item["seed"], "weighted_misses": item["weighted_misses"],
                           "small_catches": item["small_catches"], "large_catches": item["large_catches"],
                           "large_ready_departures": item["large_ready_departures"]} for item in episodes]
        print(json.dumps({"event": "condition_complete", "condition": name,
                          "mean_weighted_misses": compact[name]["mean_weighted_misses"]}), flush=True)
    controlled = [_episode(trainer, condition="nn_nn", seed=seed, external_agent=None,
                           scenario="large_hold", controller_mode="coordinated",
                           controller_settings=spec["controller"])
                  for seed in map(int, spec["controlled_hold_seeds"])]
    feasible = [item for item in controlled if item["designated_hold_outcome"] in {"caught", "missed"}]
    hold_rate = sum(item["designated_hold_outcome"] == "caught" for item in feasible) / len(feasible) if feasible else None
    reference = {"coordinated_nn_nn": "rule_rule", "coordinated_rule_player": "rule_rule",
                 "coordinated_perturb_player": "rule_perturb_player"}
    criteria = spec["criteria"]
    gates = {f"partner_regression_{name}": compact[name]["mean_weighted_misses"] <=
             compact[baseline]["mean_weighted_misses"] * (1 + float(criteria["maximum_partner_regression_fraction"]))
             for name, baseline in reference.items()}
    gates["overall_weighted_misses_relative_to_rule"] = (
        sum(float(compact[name]["mean_weighted_misses"]) for name in reference) <=
        sum(float(compact[baseline]["mean_weighted_misses"]) for baseline in reference.values()) *
        float(criteria["weighted_misses_relative_to_rule"]))
    gates["controlled_hold_95pct"] = hold_rate is not None and hold_rate >= float(criteria["minimum_feasible_hold_success_rate"])
    gates["action_chain"] = all(bool(item["action_chain_valid"]) for item in compact.values())
    report = {"version": spec["version"], "split": args.split, "source_checkpoint": str(source / spec["source_checkpoint"]),
              "actor_sha256": actor_hash, "model_qualified": False, "training_performed": False,
              "seeds": seeds, "initial_state_signatures": signatures,
              "conditions": compact, "episodes": per_seed,
              "controlled_hold": {"opportunities": len(feasible), "catches": sum(item["designated_hold_outcome"] == "caught" for item in feasible),
                                  "success_rate": hold_rate},
              "candidate_checks": {"passed": all(gates.values()), "gates": gates},
              "note": "Fixed Actor controller comparison; controlled scenes are regressions, not independent final evidence."}
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"comparison_{args.split}.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "comparison_complete", "path": str(target),
                      "candidate_checks": report["candidate_checks"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
