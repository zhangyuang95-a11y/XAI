"""Fixed-scene Pong evaluation with encounter-based success accounting."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import time
from typing import Any, Callable, TYPE_CHECKING

import numpy as np
import torch

from ..environment.engine import PongEnvironment, paddle_covers_cell, predict_next_contact
from ..policies.rule_demo import RuleDemoController
from ..policies.hybrid import LimitedPongAssist
from ..policies.coordinated import CoordinatedPongController
from .partners import ACTIONS, PerturbationPartner

if TYPE_CHECKING:
    from .runner import PongPPOTrainer


NN_CONDITIONS = ("nn_nn", "nn_rule", "nn_perturb")
BASELINE_CONDITIONS = ("idle_idle", "rule_rule")
ALL_CONDITIONS = NN_CONDITIONS + BASELINE_CONDITIONS


def _neural_actions(model: Any, trainer: "PongPPOTrainer", environment: PongEnvironment) -> tuple[int, int]:
    from domains.pong.adapters.core_adapter import feature_vector
    rows = np.stack((np.asarray(feature_vector(environment, "player"), dtype=np.float32),
                     np.asarray(feature_vector(environment, "ai"), dtype=np.float32)))
    with torch.no_grad():
        logits = model.actor_logits(torch.as_tensor(rows, dtype=torch.float32, device=trainer.device))
    player, ai = logits.argmax(-1).cpu().tolist()
    return int(player), int(ai)


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = float(sum(values) / len(values))
    return float(math.sqrt(sum((value - average) ** 2 for value in values) / len(values)))


def _initial_content_signature(environment: PongEnvironment) -> str:
    frame = environment.frame()
    body = {
        "paddles": [round(frame.player_x, 6), round(frame.ai_x, 6)],
        "balls": [(ball.ball_id, ball.active, round(ball.x, 6), round(ball.y, 6),
                   round(ball.vx, 6), round(ball.vy, 6)) for ball in frame.balls],
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _ball_target(environment: PongEnvironment, agent: str) -> tuple[str, str] | None:
    own = environment.player_x if agent == "player" else environment.ai_x
    candidates: list[tuple[float, str, str]] = []
    frame = environment.frame()
    for ball in frame.balls:
        if not ball.active or ball.is_large or ball.descending_encounter:
            continue
        prediction = predict_next_contact(ball, frame, environment.config)
        if prediction is None:
            continue
        target = prediction.contact_cells[0]
        distance = abs(own - target)
        candidates.append((prediction.time_until_contact + .01 * distance, ball.ball_id, prediction.opportunity_id))
    if not candidates:
        return None
    _, ball_id, opportunity = min(candidates)
    return ball_id, opportunity


def _large_coverage(environment: PongEnvironment) -> dict[str, bool]:
    covered: dict[str, bool] = {}
    frame = environment.frame()
    for ball in frame.balls:
        if not ball.active or not ball.is_large or ball.descending_encounter:
            continue
        prediction = predict_next_contact(ball, frame, environment.config)
        if prediction is None:
            continue
        left, right = prediction.contact_cells
        covered[prediction.opportunity_id] = (
            paddle_covers_cell(environment.player_x, left, environment.config)
            and paddle_covers_cell(environment.ai_x, right, environment.config)
        ) or (
            paddle_covers_cell(environment.player_x, right, environment.config)
            and paddle_covers_cell(environment.ai_x, left, environment.config)
        )
    return covered


def _episode(trainer: "PongPPOTrainer", *, condition: str, seed: int,
             external_agent: int | None, scenario: str = "full", controller_mode: str = "pure_nn",
             controller_settings: dict[str, Any] | None = None) -> dict[str, Any]:
    environment = PongEnvironment(trainer.env_config, seed=seed)
    if scenario != "full":
        active = {trainer.env_config.large_ball_ids[0]} if scenario == "large_hold" else set(trainer.env_config.ball_ids)
        environment.reset_training_episode(seed=seed, scenario_name="large_both_ready", active_ball_ids=active)
    initial_signature = _initial_content_signature(environment)
    designated_hold_id = (str(environment.training_scenario_metadata.get("opportunity_id"))
                          if scenario == "large_hold" else None)
    model = trainer.model
    model.eval()
    player_rule = RuleDemoController(environment.config, controlled_agent="player")
    ai_rule = RuleDemoController(environment.config, controlled_agent="ai")
    perturb = (PerturbationPartner(environment.config, external_agent, seed=seed + 17)
               if condition in {"nn_perturb", "rule_perturb"} and external_agent is not None else None)
    diagnostics: defaultdict[str, int] = defaultdict(int)
    action_chain_valid = True
    previous_targets: tuple[tuple[str, str] | None, tuple[str, str] | None] | None = None
    repeated_opportunities: set[str] = set()
    ready: dict[str, bool] = {}
    leave_recorded: set[tuple[str, str]] = set()
    ready_departures: list[dict[str, Any]] = []
    encountered: dict[str, dict[str, Any]] = {}
    assist_settings = trainer.config.get("hybrid", {})
    guards = [LimitedPongAssist(environment.config,
                               urgent_window_seconds=float(assist_settings.get("urgent_window_seconds", .6)),
                               minimum_recovery_margin_seconds=float(assist_settings.get("minimum_recovery_margin_seconds", .1)),
                               decision_frames=trainer.interval,
                               tradeoff_window_seconds=float(assist_settings.get("tradeoff_window_seconds", 1.2))) for _ in range(2)]
    coordinated = [CoordinatedPongController(environment.config, agent=agent,
                    **(controller_settings or {}))
                   for agent in ("player", "ai")]
    neural_decisions = 0
    interventions = 0
    critical_interventions = 0
    critical_decisions = 0
    rule_triggers: defaultdict[str, int] = defaultdict(int)
    intervention_streak = 0
    maximum_intervention_streak = 0
    coordinated_decision_ms: list[float] = []

    while not environment.terminal:
        if condition == "idle_idle":
            player, ai = "stay", "stay"
        elif condition == "rule_rule":
            player, ai = player_rule.choose(environment.frame()).action, ai_rule.choose(environment.frame()).action
        elif condition == "rule_perturb" and perturb is not None:
            player, ai = perturb.choose(environment), ai_rule.choose(environment.frame()).action
        else:
            player_index, ai_index = _neural_actions(model, trainer, environment)
            player, ai = ACTIONS[player_index], ACTIONS[ai_index]
            if controller_mode in {"hybrid", "coordinated"}:
                for agent_index, agent in enumerate(("player", "ai")):
                    if agent_index == external_agent:
                        continue
                    current_frame = environment.frame()
                    if any((prediction is not None and prediction.time_until_contact <=
                            float(assist_settings.get("urgent_window_seconds", .6)))
                           for ball in current_frame.balls if ball.active and not ball.descending_encounter
                           for prediction in (predict_next_contact(ball, current_frame, environment.config),)):
                        critical_decisions += 1
                    proposed = player if agent_index == 0 else ai
                    if controller_mode == "hybrid":
                        decision = guards[agent_index].choose(environment, agent, proposed)
                    else:
                        started = time.perf_counter()
                        decision = coordinated[agent_index].choose(environment, proposed)
                        coordinated_decision_ms.append((time.perf_counter() - started) * 1000)
                    neural_decisions += 1
                    if decision.intervened:
                        interventions += 1
                        intervention_streak += 1
                        maximum_intervention_streak = max(maximum_intervention_streak, intervention_streak)
                        rule_triggers[str(decision.intervention_reason)] += 1
                        critical_interventions += int(bool(decision.evidence.get("target") or
                                                           decision.evidence.get("target_ball_id")))
                    else:
                        intervention_streak = 0
                    if agent_index == 0:
                        player = decision.controller_selected_action
                    else:
                        ai = decision.controller_selected_action
            else:
                neural_decisions += 2 - int(external_agent is not None)
            if condition == "nn_rule" and external_agent is not None:
                if external_agent == 0:
                    player = player_rule.choose(environment.frame()).action
                else:
                    ai = ai_rule.choose(environment.frame()).action
            elif condition == "nn_perturb" and perturb is not None:
                if external_agent == 0:
                    player = perturb.choose(environment)
                else:
                    ai = perturb.choose(environment)
        if player not in ACTIONS or ai not in ACTIONS:
            action_chain_valid = False
            break
        targets = (_ball_target(environment, "player"), _ball_target(environment, "ai"))
        both_actually_approach = (player != "stay" and ai != "stay" and
                                  (player == "right") == (targets[0] is not None and
                                  next((ball.x for ball in environment.balls if ball.ball_id == targets[0][0]),
                                        environment.player_x) >= environment.player_x) and
                                  (ai == "right") == (targets[1] is not None and
                                  next((ball.x for ball in environment.balls if ball.ball_id == targets[1][0]),
                                        environment.ai_x) >= environment.ai_x))
        if targets[0] is not None and targets[0] == targets[1] and previous_targets == targets and both_actually_approach:
            opportunity = targets[0][1]
            if opportunity not in repeated_opportunities:
                diagnostics["repeated_small_pursuit"] += 1
                repeated_opportunities.add(opportunity)
        previous_targets = targets
        for opportunity, covered in _large_coverage(environment).items():
            if covered:
                ready[opportunity] = True
            elif ready.get(opportunity) and (opportunity, "coverage_lost") not in leave_recorded:
                diagnostics["left_large_after_ready"] += 1
                leave_recorded.add((opportunity, "coverage_lost"))
                ready_departures.append({"opportunity_id": opportunity, "outcome": None})
        for position, action in ((environment.player_x, player), (environment.ai_x, ai)):
            maximum = environment.config.width - environment.config.paddle_width
            if (position <= 1e-6 and action == "left") or (position >= maximum - 1e-6 and action == "right"):
                diagnostics["boundary_actions"] += 1
        for _ in range(trainer.interval):
            transition = environment.step(player, ai)
            if transition.player_action != player or transition.ai_action != ai:
                action_chain_valid = False
            for event in transition.events:
                if event.get("event") != "encounter":
                    continue
                encountered[str(event["encounter_id"])] = dict(event)
                for departure in ready_departures:
                    if departure["opportunity_id"] == str(event["encounter_id"]):
                        departure["outcome"] = event.get("outcome")
                if event.get("ball_kind") == "large" and event.get("outcome") == "missed":
                    coverage = event.get("coverage", {})
                    player_covered = any(coverage.get("player", ()))
                    ai_covered = any(coverage.get("ai", ()))
                    if not player_covered and not ai_covered:
                        diagnostics["large_miss_both_uncovered"] += 1
                    elif player_covered and ai_covered:
                        diagnostics["large_miss_invalid_dual_coverage"] += 1
                    else:
                        diagnostics["large_miss_one_side_covered"] += 1
            if environment.terminal:
                break

    encounters = [item for item in environment.history if item.get("event") == "encounter"]
    scored = [item for item in environment.history if item.get("event") == "miss_scored"]
    small_encounters = [item for item in encounters if item.get("ball_kind") == "small"]
    large_encounters = [item for item in encounters if item.get("ball_kind") == "large"]
    small_catches = sum(item.get("outcome") == "caught" for item in small_encounters)
    large_catches = sum(item.get("outcome") == "caught" for item in large_encounters)
    small_misses = sum(item.get("ball_kind") == "small" for item in scored)
    large_misses = sum(item.get("ball_kind") == "large" for item in scored)
    transition_opportunities = 0
    transition_catches = 0
    for index, event in enumerate(encounters):
        if event.get("outcome") != "caught":
            continue
        following = next((later for later in encounters[index + 1:]
                          if later.get("ball_id") != event.get("ball_id") and
                          float(later.get("time_seconds", 1e9)) - float(event.get("time_seconds", 0)) <= 3.0), None)
        if following is not None:
            transition_opportunities += 1
            transition_catches += int(following.get("outcome") == "caught")
    failed_but_unscored = [item for item in encounters if item.get("outcome") == "missed" and
                           str(item.get("encounter_id")) not in {str(score.get("encounter_id")) for score in scored}]
    balls: dict[str, dict[str, int]] = {ball_id: {"catches": 0, "missed_encounters": 0, "scored_misses": 0}
                                         for ball_id in trainer.env_config.ball_ids}
    for event in encounters:
        item = balls[str(event["ball_id"])]
        item["catches" if event.get("outcome") == "caught" else "missed_encounters"] += 1
    for event in scored:
        balls[str(event["ball_id"])]["scored_misses"] += 1
    return {
        "seed": seed, "initial_state_signature": initial_signature,
        "scenario": scenario, "external_agent": external_agent, "action_chain_valid": action_chain_valid,
        "weighted_misses": small_misses + 3 * large_misses,
        "small_misses": small_misses, "large_misses": large_misses,
        "small_opportunities": len(small_encounters), "large_opportunities": len(large_encounters),
        "small_catches": small_catches, "large_catches": large_catches,
        "small_success_rate": (small_catches / len(small_encounters)) if small_encounters else None,
        "large_success_rate": (large_catches / len(large_encounters)) if large_encounters else None,
        "unscored_confirmed_misses_at_terminal": len(failed_but_unscored),
        "per_ball": balls, "diagnostics": dict(diagnostics),
        "encounter_outcomes": {key: value.get("outcome") for key, value in encountered.items()},
        "large_ready_departures": ready_departures,
        "designated_hold_opportunity_id": designated_hold_id,
        "designated_hold_outcome": encountered.get(designated_hold_id, {}).get("outcome") if designated_hold_id else None,
        "controller_mode": controller_mode, "neural_decisions": neural_decisions,
        "interventions": interventions, "critical_interventions": critical_interventions,
        "critical_decisions": critical_decisions,
        "transition_after_catch_opportunities": transition_opportunities,
        "transition_after_catch_catches": transition_catches,
        "rule_triggers": dict(rule_triggers), "maximum_intervention_streak": maximum_intervention_streak,
        "coordinated_decision_ms": coordinated_decision_ms,
    }


def _summarize(condition: str, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    diagnostics: defaultdict[str, int] = defaultdict(int)
    for episode in episodes:
        for key, value in episode["diagnostics"].items():
            diagnostics[key] += int(value)
    rule_triggers: defaultdict[str, int] = defaultdict(int)
    for episode in episodes:
        for key, value in episode.get("rule_triggers", {}).items():
            rule_triggers[key] += int(value)
    small_opportunities = sum(int(item["small_opportunities"]) for item in episodes)
    large_opportunities = sum(int(item["large_opportunities"]) for item in episodes)
    small_catches = sum(int(item["small_catches"]) for item in episodes)
    large_catches = sum(int(item["large_catches"]) for item in episodes)
    per_episode_small = [float(item["small_success_rate"]) for item in episodes if item["small_success_rate"] is not None]
    per_episode_large = [float(item["large_success_rate"]) for item in episodes if item["large_success_rate"] is not None]
    roles: dict[str, dict[str, Any]] = {}
    for role, external in (("player", 1), ("ai", 0)):
        subset = [item for item in episodes if item["external_agent"] == external]
        if subset:
            roles[role] = {"episodes": len(subset), "mean_weighted_misses": _mean([float(item["weighted_misses"]) for item in subset])}
    return {
        "condition": condition, "episodes": len(episodes),
        "mean_weighted_misses": _mean([float(item["weighted_misses"]) for item in episodes]) or 0.0,
        "std_weighted_misses": _std([float(item["weighted_misses"]) for item in episodes]),
        "mean_small_misses": _mean([float(item["small_misses"]) for item in episodes]) or 0.0,
        "mean_large_misses": _mean([float(item["large_misses"]) for item in episodes]) or 0.0,
        "mean_small_success_rate": _mean(per_episode_small), "mean_large_success_rate": _mean(per_episode_large),
        "aggregate_small_success_rate": (small_catches / small_opportunities) if small_opportunities else None,
        "aggregate_large_success_rate": (large_catches / large_opportunities) if large_opportunities else None,
        "small_catches": small_catches, "small_opportunities": small_opportunities,
        "large_catches": large_catches, "large_opportunities": large_opportunities,
        "unscored_confirmed_misses_at_terminal": sum(int(item["unscored_confirmed_misses_at_terminal"]) for item in episodes),
        "diagnostics": dict(diagnostics), "by_neural_role": roles,
        "neural_decisions": sum(int(item.get("neural_decisions", 0)) for item in episodes),
        "interventions": sum(int(item.get("interventions", 0)) for item in episodes),
        "intervention_rate": (sum(int(item.get("interventions", 0)) for item in episodes) /
                              max(1, sum(int(item.get("neural_decisions", 0)) for item in episodes))),
        "critical_interventions": sum(int(item.get("critical_interventions", 0)) for item in episodes),
        "critical_decisions": sum(int(item.get("critical_decisions", 0)) for item in episodes),
        "critical_intervention_rate": (sum(int(item.get("critical_interventions", 0)) for item in episodes) /
                                       max(1, sum(int(item.get("critical_decisions", 0)) for item in episodes))),
        "transition_after_catch_opportunities": sum(int(item.get("transition_after_catch_opportunities", 0)) for item in episodes),
        "transition_after_catch_success_rate": (sum(int(item.get("transition_after_catch_catches", 0)) for item in episodes) /
                                                 max(1, sum(int(item.get("transition_after_catch_opportunities", 0)) for item in episodes))),
        "causal_intervention_gain": "not_estimated_in_episode_evaluation; see paired correction replay",
        "rule_triggers": dict(rule_triggers),
        "maximum_intervention_streak": max((int(item.get("maximum_intervention_streak", 0)) for item in episodes), default=0),
        "designated_hold_opportunities": sum(item.get("designated_hold_outcome") in {"caught", "missed"} for item in episodes),
        "designated_hold_catches": sum(item.get("designated_hold_outcome") == "caught" for item in episodes),
        "action_chain_valid": all(bool(item["action_chain_valid"]) for item in episodes), "episodes_detail": episodes,
    }


def _seeds(trainer: "PongPPOTrainer", split: str) -> list[int]:
    configured = trainer.config.get("evaluation", {}).get("validation_seeds" if split == "validation" else "final_test_seeds", ())
    seeds = list(map(int, configured))
    if not seeds:
        base = 5000 if split == "validation" else 9000
        seeds = list(range(base, base + int(trainer.config["training"].get("evaluation_episodes", 16))))
    if len(seeds) != len(set(seeds)):
        raise ValueError("fixed evaluation seed list contains duplicates")
    return seeds


def _validate_scene_isolation(trainer: "PongPPOTrainer", split: str, seeds: list[int]) -> list[str]:
    signatures = [_initial_content_signature(PongEnvironment(trainer.env_config, seed=seed)) for seed in seeds]
    if len(signatures) != len(set(signatures)):
        raise ValueError("evaluation split contains duplicate initial physical states")
    other = "final_test" if split == "validation" else "validation"
    other_signatures = {_initial_content_signature(PongEnvironment(trainer.env_config, seed=seed)) for seed in _seeds(trainer, other)}
    if set(signatures) & other_signatures:
        raise ValueError("validation and final-test scenes share physical initial states")
    trained = set(getattr(trainer, "training_initial_signatures", set()))
    if trained & set(signatures):
        raise ValueError("training and evaluation scenes share physical initial states")
    return signatures


def evaluate_trainer(trainer: "PongPPOTrainer", *, split: str, partner: str = "all",
                     progress: Callable[[dict[str, Any]], None] | None = None,
                     controller_mode: str = "pure_nn") -> dict[str, Any]:
    if controller_mode == "all":
        pure = evaluate_trainer(trainer, split=split, partner=partner, progress=progress,
                                controller_mode="pure_nn")
        hybrid = evaluate_trainer(trainer, split=split, partner=partner, progress=progress,
                                  controller_mode="hybrid")
        rule = evaluate_trainer(trainer, split=split, partner="rule_rule", progress=progress,
                                controller_mode="rule_only")
        hybrid_checks = hybrid.get("progress_checks")
        if hybrid_checks:
            hybrid_checks["gates"]["same_actor_pure_nn_progress"] = bool(pure.get("progress_checks", {}).get("passed"))
            hybrid_checks["passed"] = all(value is True for value in hybrid_checks["gates"].values())
            hybrid_checks["failed_reasons"] = [name for name, passed in hybrid_checks["gates"].items() if passed is not True]
        return {"evaluation_version": "pong-v2.2-three-modes", "split": split, "controller_mode": "all",
                "modes": {"pure_nn": pure, "hybrid": hybrid, "rule_only": rule},
                "conditions": pure["conditions"], "selection": hybrid["selection"],
                "action_chain_valid": all(item["action_chain_valid"] for item in (pure, hybrid, rule)),
                "progress_checks": {"pure_nn": pure.get("progress_checks"), "hybrid": hybrid.get("progress_checks")},
                "qualification": {"performed": False, "status": "not_formal_acceptance"}}
    if controller_mode not in {"pure_nn", "hybrid", "rule_only", "coordinated"}:
        raise ValueError("controller mode must be pure_nn, hybrid, coordinated, rule_only or all")
    if split not in {"validation", "final_test"}:
        raise ValueError("split must be validation or final_test")
    requested = (("rule_rule",) if controller_mode == "rule_only" else
                 ALL_CONDITIONS if partner == "all" else (partner,))
    if set(requested) - set(ALL_CONDITIONS):
        raise ValueError("unknown evaluation partner condition")
    seeds = _seeds(trainer, split)
    signatures = _validate_scene_isolation(trainer, split, seeds)
    total = len(requested) * len(seeds)
    completed = 0
    if progress:
        progress({"event": "evaluation_started", "split": split, "partner": partner, "episodes": total})
    conditions: dict[str, Any] = {}
    for condition_index, condition in enumerate(requested, 1):
        episodes = []
        for index, seed in enumerate(seeds):
            external = index % 2 if condition in {"nn_rule", "nn_perturb"} else None
            episodes.append(_episode(trainer, condition=condition, seed=seed, external_agent=external,
                                     controller_mode=controller_mode))
            completed += 1
            if progress and (index == 0 or index + 1 == len(seeds)):
                progress({"event": "evaluation_progress", "split": split, "condition": condition,
                          "condition_index": condition_index, "condition_count": len(requested),
                          "episodes_completed": completed, "episodes_total": total})
        conditions[condition] = _summarize(condition, episodes)
    selected = [conditions[name] for name in NN_CONDITIONS if name in conditions]
    idle = conditions.get("idle_idle")
    weighted = (_mean([float(item["mean_weighted_misses"]) for item in selected])
                if selected else float(conditions["rule_rule"]["mean_weighted_misses"]) if "rule_rule" in conditions else 0.0)
    large_rates = [float(item["aggregate_large_success_rate"]) for item in selected if item["aggregate_large_success_rate"] is not None]
    idle_score = float(idle["mean_weighted_misses"]) if idle else 0.0
    controlled: dict[str, Any] = {}
    controlled_seeds = list(map(int, trainer.config.get("evaluation", {}).get("controlled_hold_seeds", ())))
    if partner == "all" and controlled_seeds:
        controlled["large_hold"] = _summarize("large_hold", [
            _episode(trainer, condition="nn_nn", seed=seed, external_agent=None, scenario="large_hold",
                     controller_mode=controller_mode)
            for seed in controlled_seeds
        ])
    role_swap_diagnostics: dict[str, Any] = {}
    if partner == "all" and bool(trainer.config.get("evaluation", {}).get("role_swap_diagnostics", True)):
        # These paired runs are diagnostic evidence only. They deliberately do
        # not enter the historical three-condition candidate score.
        for condition in ("nn_rule", "nn_perturb"):
            by_role: dict[str, Any] = {}
            for external, label in ((0, "external_player"), (1, "external_ai")):
                by_role[label] = _summarize(condition, [
                    _episode(trainer, condition=condition, seed=seed, external_agent=external,
                             controller_mode=controller_mode)
                    for seed in seeds
                ])
            role_swap_diagnostics[condition] = by_role
    action_chain_valid = all(item["action_chain_valid"] for item in conditions.values())
    report = {
        "evaluation_version": "pong-nn-v2.2-encounter-accounting" if trainer.config.get("curriculum", {}).get("v22") else "pong-nn-v2.1-encounter-accounting",
        "controller_mode": controller_mode, "split": split, "seeds": seeds,
        "initial_state_signatures": signatures, "conditions": conditions, "controlled_checks": controlled,
        "role_swap_diagnostics": role_swap_diagnostics,
        "additional_diagnostic_episodes": sum(
            sum(int(item["episodes"]) for item in values.values())
            for values in role_swap_diagnostics.values()
        ) + sum(int(item["episodes"]) for item in controlled.values()),
        "action_chain_valid": action_chain_valid,
        "selection": {
            "mean_weighted_misses": weighted,
            "mean_large_success_rate": _mean(large_rates),
            "weighted_miss_reduction_vs_idle": ((idle_score - weighted) / idle_score) if idle_score else None,
            "selection_conditions": [item["condition"] for item in selected] if selected else ["rule_rule"],
            "rule": ("rule-only diagnostic; excluded from NN checkpoint selection" if controller_mode == "rule_only"
                     else "lower three-NN-condition mean weighted misses; then higher direct large-ball success; then earlier checkpoint"),
        }, "qualification": {"performed": False, "status": "not_run"},
    }
    trainer.model.train()
    if trainer.config.get("curriculum", {}).get("v22") and partner == "all" and controller_mode != "rule_only":
        held = controlled.get("large_hold", {})
        held_count = int(held.get("designated_hold_opportunities", 0))
        hold_success = held.get("designated_hold_catches", 0) / held_count if held_count else None
        selected = report["selection"]
        goals = trainer.config.get("evaluation", {})
        checks: dict[str, bool | None] = {
            "combined_weighted_misses": float(selected["mean_weighted_misses"]) <= float(goals.get("pure_combined_miss_limit", 38.925)),
            "nn_nn_small_success": (conditions["nn_nn"]["aggregate_small_success_rate"] is not None and
                                    conditions["nn_nn"]["aggregate_small_success_rate"] >= float(goals.get("minimum_small_success_rate", .45))),
            "nn_nn_large_success": (conditions["nn_nn"]["aggregate_large_success_rate"] is not None and
                                    conditions["nn_nn"]["aggregate_large_success_rate"] >= float(goals.get("minimum_large_success_rate", .50))),
            "designated_hold_success": hold_success is not None and hold_success >= float(goals.get("minimum_hold_success_rate", .90)),
            "action_chain": bool(report["action_chain_valid"]),
        }
        for name, parent in goals.get("parent_weighted_misses", {}).items():
            checks[f"no_regression_{name}"] = (float(conditions[name]["mean_weighted_misses"]) <=
                                                float(parent) * (1 + float(goals.get("maximum_partner_regression_fraction", .05))))
        if controller_mode == "hybrid":
            nn_cases = [conditions[name] for name in NN_CONDITIONS]
            intervention_rate = sum(item["interventions"] for item in nn_cases) / max(1, sum(item["neural_decisions"] for item in nn_cases))
            checks["nn_action_fraction_at_least_80pct"] = intervention_rate <= float(goals.get("maximum_hybrid_intervention_fraction", .20))
            checks["no_sustained_takeover"] = max(item["maximum_intervention_streak"] for item in nn_cases) <= int(goals.get("maximum_consecutive_interventions", 6))
            report["intervention_rate"] = intervention_rate
            report["intervention_rate_target"] = float(goals.get("maximum_hybrid_intervention_fraction", .20))
        elif controller_mode == "coordinated":
            nn_cases = [conditions[name] for name in NN_CONDITIONS]
            report["intervention_rate"] = sum(item["interventions"] for item in nn_cases) / max(
                1, sum(item["neural_decisions"] for item in nn_cases))
            checks = {"action_chain": bool(report["action_chain_valid"]),
                      "designated_hold_success": hold_success is not None and hold_success >= .95}
            report["coordination_checks"] = {"gates": checks, "passed": all(checks.values()),
                                             "status": "candidate_diagnostics_only"}
        report["controlled_checks"]["designated_large_hold"] = {
            "opportunities": held_count, "catches": held.get("designated_hold_catches", 0),
            "success_rate": hold_success, "only_first_preselected_opportunity": True,
        }
        report["progress_checks"] = {"passed": all(value is True for value in checks.values()),
                                     "gates": checks,
                                     "failed_reasons": [name for name, passed in checks.items() if passed is not True],
                                     "status": "training_progress_only"}
    if progress:
        progress({"event": "evaluation_complete", "split": split, "partner": partner,
                  "episodes_completed": completed, "episodes_total": total})
    return report
