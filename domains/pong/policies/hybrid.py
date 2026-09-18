"""Small, auditable corrections to a Pong Actor's proposed action.

The controller sees a public pre-action frame.  It never assumes a player's
unsubmitted command, and never chooses a ball as a general-purpose planner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..config import PongConfig
from ..environment.engine import PongEnvironment, paddle_covers_cell, paddle_target_left, predict_next_contact

ACTIONS = ("left", "right", "stay")
VERSION = "pong-limited-assist.v2.2"


@dataclass(frozen=True)
class HybridDecision:
    nn_proposed_action: str
    controller_selected_action: str
    intervention_reason: str | None
    evidence: dict[str, Any]
    rule_version: str = VERSION

    @property
    def intervened(self) -> bool:
        return self.nn_proposed_action != self.controller_selected_action

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "intervened": self.intervened}


class LimitedPongAssist:
    """Only correct a provable near-contact failure or an ineffective edge move."""

    def __init__(self, config: PongConfig, *, urgent_window_seconds: float = .6,
                 minimum_recovery_margin_seconds: float = .1, decision_frames: int = 6,
                 tradeoff_window_seconds: float = 1.2) -> None:
        self.config = config
        self.window = float(urgent_window_seconds)
        self.margin = float(minimum_recovery_margin_seconds)
        self.tradeoff_window = max(self.window, float(tradeoff_window_seconds))
        self.decision_frames = int(decision_frames)
        self.commitments: dict[str, dict[str, Any]] = {}
        self.last_release: dict[str, str] = {}

    def reset(self) -> None:
        self.commitments.clear()
        self.last_release.clear()

    def state_dict(self) -> dict[str, Any]:
        return {"rule_version": VERSION, "commitments": dict(self.commitments),
                "last_release": dict(self.last_release)}

    def restore_state(self, state: Mapping[str, Any]) -> None:
        if state.get("rule_version") != VERSION:
            raise ValueError("hybrid rule version mismatch")
        self.commitments = {str(k): dict(v) for k, v in dict(state.get("commitments", {})).items()}
        self.last_release = {str(k): str(v) for k, v in dict(state.get("last_release", {})).items()}

    def choose(self, env: PongEnvironment, agent: str, proposal: str) -> HybridDecision:
        if agent not in {"player", "ai"} or proposal not in ACTIONS:
            raise ValueError("unknown agent or proposed action")
        frame = env.frame()
        own = frame.player_x if agent == "player" else frame.ai_x
        other = frame.ai_x if agent == "player" else frame.player_x
        maximum = self.config.width - self.config.paddle_width
        duration = self.decision_frames * self.config.fixed_dt
        step = self.config.paddle_speed_per_second * duration

        candidates: list[dict[str, Any]] = []
        valid_opportunities: set[str] = set()
        for ball in frame.balls:
            if not ball.active or ball.descending_encounter:
                continue
            prediction = predict_next_contact(ball, frame, self.config)
            if prediction is None:
                continue
            valid_opportunities.add(prediction.opportunity_id)
            if prediction.time_until_contact > self.tradeoff_window:
                continue
            cells = prediction.contact_cells
            if len(cells) == 1:
                side = "single"
                if paddle_covers_cell(other, cells[0], self.config):
                    # There is no reason to take a small ball already covered
                    # by the other paddle under a static public-state check.
                    continue
                target = cells[0]
            else:
                left, right = cells
                other_left = paddle_covers_cell(other, left, self.config)
                other_right = paddle_covers_cell(other, right, self.config)
                if other_left and not other_right:
                    side, target = "right", right
                elif other_right and not other_left:
                    side, target = "left", left
                else:
                    # An assumption that the other paddle will move is not
                    # proof of a successful cooperative catch.
                    continue
            target_left = paddle_target_left(own, target, self.config)
            slack = prediction.time_until_contact - abs(target_left - own) / self.config.paddle_speed_per_second
            if slack < self.margin and not paddle_covers_cell(own, target, self.config):
                continue
            candidates.append({"ball_id": ball.ball_id, "opportunity_id": prediction.opportunity_id,
                               "ball_kind": "large" if ball.is_large else "small", "side": side,
                               "target_cell": float(target), "time_until_contact": float(prediction.time_until_contact),
                               "target_left": float(target_left), "slack": float(slack),
                               "weight": 3 if ball.is_large else 1})

        old = self.commitments.get(agent)
        if old and old["opportunity_id"] not in valid_opportunities:
            self.last_release[agent] = "opportunity_ended"
            self.commitments.pop(agent, None)
        if old and not any(c["opportunity_id"] == old["opportunity_id"] and c["side"] == old["side"] for c in candidates):
            self.last_release[agent] = "coverage_or_partner_changed"
            self.commitments.pop(agent, None)

        if (own <= 1e-6 and proposal == "left") or (own >= maximum - 1e-6 and proposal == "right"):
            return HybridDecision(proposal, "stay", "boundary_no_motion",
                                  {"agent": agent, "position": own, "boundary": 0.0 if proposal == "left" else maximum,
                                   "excluded_future_player_action": True})

        if not candidates:
            return HybridDecision(proposal, proposal, None,
                                  {"agent": agent, "reason": "no_proven_near_contact_correction",
                                   "released": self.last_release.get(agent)})
        if not any(item["time_until_contact"] <= self.window for item in candidates):
            return HybridDecision(proposal, proposal, None,
                                  {"agent": agent, "reason": "no_urgent_verified_intervention"})

        # Rank by penalty at risk, then urgency.  Evaluate all three first
        # actions; preserve the Actor unless an alternative strictly protects
        # more currently feasible opportunities.  No imagined partner move.
        candidates.sort(key=lambda c: (-c["weight"], c["time_until_contact"], c["ball_id"]))
        positions = {action: max(0.0, min(maximum, own + (step if action == "right" else -step if action == "left" else 0.0)))
                     for action in ACTIONS}

        def protected(action: str, candidate: dict[str, Any]) -> bool:
            contact_time = candidate["time_until_contact"]
            if contact_time <= duration + 1e-9:
                delta = self.config.paddle_speed_per_second * max(0.0, contact_time)
                at_contact = max(0.0, min(maximum, own + (delta if action == "right" else
                                                           -delta if action == "left" else 0.0)))
                return paddle_covers_cell(at_contact, candidate["target_cell"], self.config)
            position = positions[action]
            remaining = contact_time - duration
            residual = abs(paddle_target_left(position, candidate["target_cell"], self.config) - position)
            return residual / self.config.paddle_speed_per_second <= remaining - self.margin + 1e-6

        ordered = sorted(candidates, key=lambda c: (c["time_until_contact"], -c["weight"], c["ball_id"]))

        def best_sequence(action: str) -> tuple[int, list[dict[str, Any]]]:
            def visit(index: int, position: float, available_at: float,
                      chosen: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
                if index == len(ordered):
                    return sum(item["weight"] for item in chosen), chosen
                skipped = visit(index + 1, position, available_at, chosen)
                candidate = ordered[index]
                contact_at = candidate["time_until_contact"]
                if contact_at <= duration + 1e-9:
                    if not protected(action, candidate):
                        return skipped
                    next_position = position
                    next_time = available_at
                else:
                    next_position = paddle_target_left(position, candidate["target_cell"], self.config)
                    arrival = available_at + abs(next_position - position) / self.config.paddle_speed_per_second
                    if arrival > contact_at - self.margin + 1e-6:
                        return skipped
                    next_time = contact_at
                taken = visit(index + 1, next_position, next_time, chosen + [candidate])
                return taken if taken[0] > skipped[0] else skipped
            return visit(0, positions[action], duration, [])

        plans = {action: best_sequence(action) for action in ACTIONS}
        weighted = {action: plan[0] for action, plan in plans.items()}
        proposed_score = weighted[proposal]
        alternatives = [action for action in ACTIONS if weighted[action] > proposed_score]
        if not alternatives:
            return HybridDecision(proposal, proposal, None,
                                  {"agent": agent, "candidate_scores": weighted,
                                   "reason": "proposal_preserves_feasible_opportunities"})
        alternatives.sort(key=lambda action: (-weighted[action],
                                               sum(abs(positions[action] - c["target_left"]) for c in candidates),
                                               action != "stay"))
        chosen = alternatives[0]
        prior_ids = {item["opportunity_id"] for item in plans[proposal][1]}
        saved = next((item for item in plans[chosen][1] if item["opportunity_id"] not in prior_ids), None)
        if saved is None:
            return HybridDecision(proposal, proposal, None, {"agent": agent, "reason": "ambiguous_correction"})
        reason = "large_side_rescue" if saved["ball_kind"] == "large" else "near_miss_rescue"
        if paddle_covers_cell(own, saved["target_cell"], self.config) and chosen == "stay":
            reason = "protect_existing_coverage"
        if saved["ball_kind"] == "large":
            self.commitments[agent] = {"ball_id": saved["ball_id"], "opportunity_id": saved["opportunity_id"],
                                       "side": saved["side"], "created_frame": frame.frame,
                                       "last_validated_frame": frame.frame, "release_reason": None}
        return HybridDecision(proposal, chosen, reason,
                              {"agent": agent, "candidate_scores": weighted, "target": saved,
                               "other_position": other, "decision_seconds": duration,
                               "excluded_future_player_action": True})
