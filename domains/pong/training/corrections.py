"""Separate hybrid trajectories and paired, short-horizon correction checks.

Nothing collected here is a PPO transition.  Each comparison starts from the
same environment snapshot; both continuations use the same frozen Actor.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Mapping

import numpy as np
import torch

from ..adapters.core_adapter import feature_vector
from ..config import PongConfig
from ..environment.engine import PongEnvironment
from ..policies.hybrid import ACTIONS, LimitedPongAssist, VERSION


def _miss_weight(events: list[dict[str, Any]]) -> int:
    # A contact that has already failed is a loss in this short comparison
    # even if its bottom-edge score occurs just after the replay horizon.
    losses: dict[str, int] = {}
    for event in events:
        if event.get("event") == "encounter" and event.get("outcome") == "missed":
            losses[str(event["encounter_id"])] = 3 if event.get("ball_kind") == "large" else 1
        elif event.get("event") == "miss_scored":
            losses[str(event["encounter_id"])] = 3 if event.get("ball_kind") == "large" else 1
    return sum(losses.values())


class CorrectionCollector:
    """One independent sampling stream, split by entire episode identity."""

    def __init__(self, config: PongConfig, settings: Mapping[str, Any], *, interval: int, seed: int) -> None:
        self.config = config
        self.settings = dict(settings)
        self.interval = interval
        self.seed = seed
        self.environment = PongEnvironment(config, seed=seed)
        self.scenarios = ("large_both_ready", "large_one_holds", "large_both_approach",
                          "dual_small", "small_then_large", "dual_large", "full")
        self.scenario_cursor = 0
        self.guards = [LimitedPongAssist(config, decision_frames=interval,
                                        urgent_window_seconds=float(settings.get("urgent_window_seconds", .6)),
                                        minimum_recovery_margin_seconds=float(settings.get("minimum_recovery_margin_seconds", .1)),
                                        tradeoff_window_seconds=float(settings.get("tradeoff_window_seconds", 1.2)))
                       for _ in range(2)]
        self.episode_id = 1_000_000
        self.episode_decisions = 0
        self.assisted_steps = 0
        self.comparison_steps = 0
        self.interventions = 0
        self.supervised_updates = 0
        self.accepted: list[dict[str, Any]] = []
        self.rejected: list[dict[str, Any]] = []
        self.reason_counts: Counter[str] = Counter()
        self.settled_updates: list[dict[str, Any]] = []
        self._sample_rng = np.random.default_rng(seed + 37)
        self._reset_episode()

    def _reset_episode(self) -> None:
        scenario = self.scenarios[self.scenario_cursor % len(self.scenarios)]
        self.scenario_cursor += 1
        if scenario.startswith("large_"):
            active = {self.config.large_ball_ids[0]}
        elif scenario == "dual_large":
            active = set(self.config.large_ball_ids)
        elif scenario == "dual_small":
            active = set(self.config.small_ball_ids[:2])
        elif scenario == "small_then_large":
            active = {self.config.small_ball_ids[0], self.config.large_ball_ids[0]}
        else:
            active = set(self.config.ball_ids)
        self.environment.reset_training_episode(seed=self.seed + self.episode_id,
                                                scenario_name=scenario, active_ball_ids=active)
        for guard in self.guards:
            guard.reset()

    def _actor(self, model: torch.nn.Module, device: torch.device, env: PongEnvironment) -> tuple[np.ndarray, list[int]]:
        rows = np.stack((feature_vector(env, "player"), feature_vector(env, "ai"))).astype(np.float32)
        with torch.no_grad():
            probabilities = torch.softmax(model.actor_logits(torch.as_tensor(rows, device=device)), -1).cpu().numpy()
        return probabilities, probabilities.argmax(axis=1).tolist()

    def _step(self, env: PongEnvironment, actions: list[int]) -> int:
        count = 0
        for _ in range(self.interval):
            env.step(ACTIONS[actions[0]], ACTIONS[actions[1]])
            count += 1
            if env.terminal:
                break
        return count

    def _compare(self, snapshot: Mapping[str, Any], original: list[int], selected: list[int],
                 target: str, model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
        outcomes: list[dict[str, Any]] = []
        for first in (original, selected):
            env = PongEnvironment(self.config, seed=self.seed)
            env.restore(deepcopy(snapshot))
            start = len(env.history)
            decisions_taken = 1
            self._step(env, first)
            decisions = max(1, round(8.0 / (self.interval * self.config.fixed_dt)))
            for _ in range(decisions - 1):
                if env.terminal:
                    break
                _, actions = self._actor(model, device, env)
                self._step(env, actions)
                decisions_taken += 1
            new_events = [dict(event) for event in env.history[start:]]
            self.comparison_steps += decisions_taken
            target_events = [event for event in new_events if str(event.get("encounter_id")) == target]
            score = _miss_weight(new_events)
            other_losses = {str(event.get("encounter_id")): event for event in new_events
                            if str(event.get("encounter_id")) != target and
                            ((event.get("event") == "encounter" and event.get("outcome") == "missed")
                             or event.get("event") == "miss_scored")}
            outcomes.append({"weighted_misses": score,
                             "target_encounter": [event for event in target_events if event.get("event") == "encounter"],
                             "target_scored_miss": any(event.get("event") == "miss_scored" for event in target_events),
                             "other_ball_losses": list(other_losses.values())})
        base, corrected = outcomes
        gain = base["weighted_misses"] - corrected["weighted_misses"]
        settled = bool(base["target_encounter"] and corrected["target_encounter"])
        return {"original": base, "corrected": corrected, "net_weighted_gain": gain,
                "status": "accepted" if settled and gain >= float(self.settings.get("minimum_comparison_gain", 1.0))
                else "rejected_unsettled" if not settled else "rejected_no_net_gain"}

    def collect(self, model: torch.nn.Module, device: torch.device, decisions: int,
                *, ppo_joint_steps: int = 0) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for _ in range(decisions):
            env = self.environment
            before = env.snapshot()
            observations = np.stack((feature_vector(env, "player"), feature_vector(env, "ai"))).astype(np.float32)
            probabilities, original = self._actor(model, device, env)
            selected = list(original)
            interventions: list[tuple[int, dict[str, Any]]] = []
            for index, agent in enumerate(("player", "ai")):
                decision = self.guards[index].choose(env, agent, ACTIONS[original[index]])
                selected[index] = ACTIONS.index(decision.controller_selected_action)
                if decision.intervened:
                    interventions.append((index, decision.to_dict()))
            self._step(env, selected)
            new_events = [dict(event) for event in env.history[len(before["history"]):]]
            for prior in (*self.accepted, *self.rejected):
                if int(prior["episode_id"]) != self.episode_id:
                    continue
                target_id = str(prior.get("candidate_evidence", {}).get("target", {}).get("opportunity_id", ""))
                for event in new_events:
                    if event.get("event") == "encounter" and str(event.get("encounter_id")) == target_id:
                        prior["actual_encounter_outcomes"].append(event)
                        self.settled_updates.append({"episode_id": prior["episode_id"], "sample_frame": prior["frame"],
                                                     "encounter_id": target_id, "event": event})
                    if event.get("event") == "miss_scored" and str(event.get("encounter_id")) != target_id:
                        prior.setdefault("actual_other_ball_losses", []).append(event)
            self.assisted_steps += 1
            self.episode_decisions += 1
            for index, decision in interventions:
                self.interventions += 1
                reason = str(decision["intervention_reason"])
                self.reason_counts[reason] += 1
                target = str(decision.get("evidence", {}).get("target", {}).get("opportunity_id", ""))
                isolated = list(original)
                isolated[index] = selected[index]
                comparison = self._compare(before, original, isolated, target, model, device) if target and len(interventions) == 1 else {
                    "status": "rejected_joint_intervention_attribution" if len(interventions) > 1 else "rejected_no_target_encounter",
                    "net_weighted_gain": None,
                }
                sample = {"source": "limited_rule_assist", "episode_id": self.episode_id,
                          "collected_at_ppo_joint_steps": int(ppo_joint_steps),
                          "frame": int(before["frame_index"]), "agent": ("player", "ai")[index],
                          "observation": observations[index].tolist(),
                          "nn_action_probabilities": probabilities[index].tolist(),
                          "nn_proposed_action": ACTIONS[original[index]],
                          "controller_selected_action": ACTIONS[selected[index]],
                          "intervention_reason": reason, "rule_version": VERSION,
                          "candidate_evidence": decision["evidence"],
                          "actual_encounter_outcomes": [dict(event) for event in new_events
                                                         if event.get("event") == "encounter"],
                          "actual_other_ball_losses": [dict(event) for event in new_events
                                                       if event.get("event") == "miss_scored" and
                                                       str(event.get("encounter_id")) != target],
                          "other_ball_losses": comparison.get("corrected", {}).get("other_ball_losses", []),
                          "teacher_validation_status": comparison["status"], "comparison": comparison,
                          "split": "validation" if self.episode_id % 5 == 0 else "train"}
                for event in sample["actual_encounter_outcomes"]:
                    if str(event.get("encounter_id")) == target:
                        self.settled_updates.append({"episode_id": sample["episode_id"],
                                                     "sample_frame": sample["frame"],
                                                     "encounter_id": target, "event": event})
                records.append(sample)
                cache = self.accepted if comparison["status"] == "accepted" else self.rejected
                cache.append(sample)
                cache[:] = cache[-int(self.settings.get("correction_cache_size", 20_000)):]
            short_limit = round(6.0 / (self.interval * self.config.fixed_dt))
            if env.terminal or (self.episode_decisions >= short_limit
                                and not any(ball.pending_miss for ball in env.balls)):
                self.episode_id += 1
                self.episode_decisions = 0
                self._reset_episode()
        return records

    def training_batch(self, count: int, *, current_ppo_joint_steps: int = 0) -> tuple[np.ndarray, np.ndarray] | None:
        max_age = int(self.settings.get("correction_max_age_joint_steps", 50_000))
        rows = [item for item in self.accepted if item["split"] == "train" and
                current_ppo_joint_steps - int(item.get("collected_at_ppo_joint_steps", 0)) <= max_age]
        if not rows:
            return None
        indices = self._sample_rng.choice(len(rows), size=min(count, len(rows)), replace=False)
        return (np.asarray([rows[int(index)]["observation"] for index in indices], dtype=np.float32),
                np.asarray([ACTIONS.index(rows[int(index)]["controller_selected_action"])
                            for index in indices], dtype=np.int64))

    def state_dict(self) -> dict[str, Any]:
        return {"environment": self.environment.snapshot(), "guards": [item.state_dict() for item in self.guards],
                "episode_id": self.episode_id, "episode_decisions": self.episode_decisions,
                "scenario_cursor": self.scenario_cursor,
                "assisted_steps": self.assisted_steps, "comparison_steps": self.comparison_steps,
                "interventions": self.interventions, "supervised_updates": self.supervised_updates,
                "accepted": self.accepted, "rejected": self.rejected, "reason_counts": dict(self.reason_counts),
                "settled_updates": self.settled_updates,
                "sample_rng": self._sample_rng.bit_generator.state}

    def restore_state(self, payload: Mapping[str, Any]) -> None:
        self.environment.restore(payload["environment"])
        for guard, state in zip(self.guards, payload["guards"]):
            guard.restore_state(state)
        self.episode_id = int(payload["episode_id"])
        self.episode_decisions = int(payload["episode_decisions"])
        self.scenario_cursor = int(payload["scenario_cursor"])
        self.assisted_steps = int(payload["assisted_steps"])
        self.comparison_steps = int(payload["comparison_steps"])
        self.interventions = int(payload["interventions"])
        self.supervised_updates = int(payload["supervised_updates"])
        self.accepted = [dict(item) for item in payload["accepted"]]
        self.rejected = [dict(item) for item in payload["rejected"]]
        self.reason_counts = Counter(payload["reason_counts"])
        self.settled_updates = [dict(item) for item in payload.get("settled_updates", [])]
        self._sample_rng.bit_generator.state = payload["sample_rng"]
