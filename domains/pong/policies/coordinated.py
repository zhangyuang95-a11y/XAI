"""Versioned Pong coordination around a frozen neural action proposal.

The rule controller owns the catch assignment.  The neural action remains
visible and is used whenever it agrees with the feasible assignment.  Nothing
here reads the other agent's action for the current decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import PongConfig
from ..environment.engine import PongEnvironment, paddle_covers_cell
from .rule_demo import RuleDemoController, _Candidate, _Commitment

VERSION = "pong-coordinated.v2.3"
_EPSILON = 1e-6


@dataclass(frozen=True)
class CoordinatedDecision:
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


class CoordinatedPongController:
    """Plan up to three upcoming encounters; retain a feasible catch pledge."""

    def __init__(self, config: PongConfig, *, agent: str = "ai", horizon_seconds: float = 10.0,
                 max_opportunities: int = 3, switch_gain: float = 1.5) -> None:
        if agent not in {"player", "ai"}:
            raise ValueError("agent must be player or ai")
        self.config = config
        self.agent = agent
        self.rule = RuleDemoController(config, controlled_agent=agent)
        self.horizon_seconds = float(horizon_seconds)
        self.max_opportunities = int(max_opportunities)
        self.switch_gain = float(switch_gain)
        self.small_commitment: str | None = None

    def reset(self) -> None:
        self.rule.reset()
        self.small_commitment = None

    def state_dict(self) -> dict[str, Any]:
        return {"version": VERSION, "rule": self.rule.state_dict(),
                "small_commitment": self.small_commitment}

    def restore_state(self, state: dict[str, Any]) -> None:
        if state.get("version") != VERSION:
            raise ValueError("coordinated controller version mismatch")
        self.rule.restore_state(state["rule"])
        self.small_commitment = state.get("small_commitment")

    def _plan(self, candidates: list[_Candidate], position: float, *, first: str | None = None) -> tuple[float, list[_Candidate]]:
        speed = self.config.paddle_speed_per_second
        margin = 0.02

        def visit(index: int, at: float, x: float, chosen: list[_Candidate]) -> tuple[float, list[_Candidate]]:
            if index == len(candidates):
                return sum(item.penalty for item in chosen), chosen
            candidate = candidates[index]
            skipped = ((-1.0, []) if index == 0 and first is not None else
                       visit(index + 1, at, x, chosen))
            # A paddle cannot leave a preceding catch before that encounter.
            arrival = at + abs(candidate.target_x - x) / speed
            contact = candidate.prediction.time_until_contact
            if arrival > contact - margin + _EPSILON:
                return skipped
            taken = visit(index + 1, contact, candidate.target_x, chosen + [candidate])
            if taken[0] > skipped[0]:
                return taken
            return skipped

        return visit(0, 0.0, position, [])

    def choose(self, env: PongEnvironment, proposal: str) -> CoordinatedDecision:
        if proposal not in {"left", "right", "stay"}:
            raise ValueError("unknown neural action")
        frame = env.frame()
        own = frame.ai_x if self.agent == "ai" else frame.player_x
        other = frame.player_x if self.agent == "ai" else frame.ai_x
        old_large = self.rule.state_dict().get("commitment")
        base = self.rule.choose(frame)
        all_candidates = self.rule._candidates(frame)
        available = [candidate for candidate in all_candidates
                     if candidate.viable and not candidate.handoff and
                     candidate.prediction.time_until_contact <= self.horizon_seconds]
        available.sort(key=lambda item: (item.prediction.time_until_contact, -item.penalty,
                                         item.prediction.ball_id))
        available = available[:self.max_opportunities]
        selected = next((item for item in all_candidates if item.prediction.opportunity_id == base.opportunity_id
                         and item.contact_side == base.contact_side), None)
        if selected is not None and selected.requires_partner and not selected.viable:
            self.rule._commitment = None
            old_large = None
            selected = None
        prior_small = next((item for item in available if item.prediction.opportunity_id == self.small_commitment), None)
        if prior_small is not None:
            selected = prior_small
        elif self.small_commitment:
            self.small_commitment = None

        best_score, best_sequence = self._plan(available, own)
        base_score = 0.0
        base_sequence: list[_Candidate] = []
        if selected is not None:
            baseline = [selected] + [item for item in available if item != selected and
                                      item.prediction.time_until_contact > selected.prediction.time_until_contact]
            base_score, base_sequence = self._plan(baseline, own, first=selected.prediction.opportunity_id)
        planned = best_sequence[0] if best_sequence else None
        switched = False
        if (planned is not None and selected is not None and not old_large
                and planned.prediction.opportunity_id != selected.prediction.opportunity_id
                and best_score >= base_score + self.switch_gain):
            selected = planned
            switched = True
        elif selected is None and planned is not None and not old_large:
            selected = planned
        actual_sequence = best_sequence if switched or (selected is planned and not base_sequence) else base_sequence

        if selected is not None:
            if selected.requires_partner:
                self.rule._commitment = _Commitment(selected.prediction.ball_id,
                                                     selected.prediction.opportunity_id,
                                                     selected.contact_side)
                self.small_commitment = None
            else:
                self.rule._commitment = None
                self.small_commitment = selected.prediction.opportunity_id
            target_cell = (selected.prediction.contact_cells[0] if selected.contact_side != "right"
                           else selected.prediction.contact_cells[-1])
            difference = selected.target_x - own
            # The environment uses a strict right-edge coverage test. A target
            # only microns away is still a real gap; do not call it a hold.
            covered_now = paddle_covers_cell(own, target_cell, self.config)
            rule_action = ("right" if difference > 0 else "left" if difference < 0 else "stay") if not covered_now else (
                "right" if difference > .025 else "left" if difference < -.025 else "stay")
            holding = paddle_covers_cell(own, target_cell, self.config) and rule_action == "stay"
            partner_cell = (selected.prediction.contact_cells[-1] if selected.contact_side == "left"
                            else selected.prediction.contact_cells[0]) if selected.requires_partner else target_cell
            partner_covered = paddle_covers_cell(other, partner_cell, self.config)
            partner_status = ("covered" if partner_covered else "reachable" if selected.viable else "unreachable")
            reason = ("keep_large_side" if holding and selected.requires_partner else
                      "keep_small_coverage" if holding else "sequence_gain" if switched else
                      "prepare_large_side" if selected.requires_partner else "approach_small")
        else:
            rule_action = base.action
            reason = "no_feasible_assignment"
            holding = False
            partner_status = "unknown"

        maximum = self.config.width - self.config.paddle_width
        boundary = (own <= _EPSILON and proposal == "left") or (own >= maximum - _EPSILON and proposal == "right")
        if selected is None and boundary:
            rule_action = "stay"
            reason = "boundary_no_motion"
        # With no verified job, a legal neural action is safe to retain.
        if selected is None and not boundary:
            rule_action = proposal
        chosen = rule_action
        candidate_rows = []
        for item in all_candidates:
            in_window = item in available
            forced_score = None
            if in_window:
                sequence = [item] + [later for later in available if later != item and
                                     later.prediction.time_until_contact > item.prediction.time_until_contact]
                forced_score = self._plan(sequence, own, first=item.prediction.opportunity_id)[0]
            candidate_rows.append(item.public() | {"weight": item.penalty,
                "in_planning_window": in_window, "plan_score_if_first": forced_score})
        evidence = {
            "version": VERSION, "agent": self.agent, "decision_frame": frame.frame,
            "target_ball_id": selected.prediction.ball_id if selected else None,
            "opportunity_id": selected.prediction.opportunity_id if selected else None,
            "target_kind": selected.prediction.ball_kind if selected else None,
            "contact_side": selected.contact_side if selected else None,
            "target_x": selected.target_x if selected else None,
            "self_distance": selected.ai_distance if selected else None,
            "partner_distance": selected.player_distance if selected else None,
            "self_reachable": selected.ai_viable if selected else False,
            "partner_status": partner_status,
            "requires_partner": selected.requires_partner if selected else False,
            "holding": holding,
            "reason": reason, "candidate_scores": {"planned": best_score, "baseline": base_score},
            "alternatives": candidate_rows, "planned_sequence": [item.prediction.ball_id for item in actual_sequence],
            "discarded_neural_action": proposal if chosen != proposal else None,
            "other_current_position": other,
            "excluded_future_player_action": True,
        }
        return CoordinatedDecision(proposal, chosen, reason if chosen != proposal else None, evidence)
