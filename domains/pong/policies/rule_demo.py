from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import PongConfig
from ..environment.engine import (
    PredictedContact,
    paddle_covers_cell,
    paddle_target_left,
    predict_next_contact,
)
from ..environment.model import PongFrame

_EPSILON = 1e-6


@dataclass(frozen=True)
class ControllerDecision:
    """Factual rule-controller decision for one authoritative state."""

    action: str
    reason: str
    intent_type: str = "no_urgent"
    target_x: float | None = None
    target_ball_id: str | None = None
    target_kind: str | None = None
    contact_side: str | None = None
    distance_cells: float | None = None
    eta_updates: int | None = None
    time_until_contact: float | None = None
    requires_partner: bool = False
    opportunity_id: str | None = None
    commitment_state: str = "none"
    candidates: tuple[dict[str, Any], ...] = ()
    conflict_ball_ids: tuple[str, ...] = ()
    player_distance_cells: float | None = None
    player_closest_ball_id: str | None = None
    player_closest_distance_cells: float | None = None
    small_first_ball_id: str | None = None
    small_first_feasible: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy() | {"candidates": [dict(item) for item in self.candidates]}


@dataclass(frozen=True)
class _Candidate:
    prediction: PredictedContact
    target_x: float
    contact_side: str | None
    requires_partner: bool
    ai_distance: float
    player_distance: float
    viable: bool
    ai_viable: bool
    handoff: bool
    reason: str

    @property
    def penalty(self) -> int:
        return 3 if self.prediction.ball_kind == "large" else 1

    def public(self) -> dict[str, Any]:
        return {
            "ball_id": self.prediction.ball_id,
            "kind": self.prediction.ball_kind,
            "opportunity_id": self.prediction.opportunity_id,
            "time_until_contact": round(self.prediction.time_until_contact, 4),
            "target_x": round(self.target_x, 4),
            "contact_side": self.contact_side,
            "ai_distance": round(self.ai_distance, 4),
            "player_distance": round(self.player_distance, 4),
            "viable": self.viable,
            "ai_viable": self.ai_viable,
            "handoff": self.handoff,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class _Commitment:
    ball_id: str
    opportunity_id: str
    contact_side: str | None


class RuleDemoController:
    """A transparent, state-only baseline with stable large-ball commitments.

    It never reads the player's unsubmitted action.  A large-ball commitment
    is bound to a ball and its particular downward opportunity, rather than to
    a fragile exact target coordinate, so a small prediction adjustment cannot
    make Robot 2 leave before the actual catch has been settled.
    """

    version = "rule-demo-continuous-24x14-v7"

    def __init__(self, config: PongConfig | None = None, *, controlled_agent: str = "ai") -> None:
        self.config = config or PongConfig()
        if controlled_agent not in {"player", "ai"}:
            raise ValueError("controlled_agent must be 'player' or 'ai'")
        self.controlled_agent = controlled_agent
        self._commitment: _Commitment | None = None

    def reset(self) -> None:
        self._commitment = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "controlled_agent": self.controlled_agent,
            "commitment": self._commitment.__dict__.copy() if self._commitment else None,
        }

    def restore_state(self, payload: dict[str, Any]) -> None:
        if payload.get("controlled_agent", self.controlled_agent) != self.controlled_agent:
            raise ValueError("rule-controller role mismatch")
        commitment = payload.get("commitment")
        self._commitment = _Commitment(**commitment) if commitment else None

    def _own_x(self, frame: PongFrame) -> float:
        return frame.ai_x if self.controlled_agent == "ai" else frame.player_x

    def _other_x(self, frame: PongFrame) -> float:
        return frame.player_x if self.controlled_agent == "ai" else frame.ai_x

    def choose(self, frame: PongFrame) -> ControllerDecision:
        if frame.terminal:
            return ControllerDecision("stay", "本局已经结束。")
        candidates = self._candidates(frame)
        public = tuple(candidate.public() for candidate in candidates)
        conflict = self._conflicts(candidates)
        player_closest = min(candidates, key=lambda item: item.player_distance, default=None)

        committed = self._committed_candidate(candidates)
        if committed is not None:
            return self._decision(
                frame, committed, public, conflict, player_closest,
                commitment_state="waiting_for_large" if committed.ai_distance <= 0.05 else "approaching_large",
                candidates=candidates,
            )

        available = [item for item in candidates if item.viable and not item.handoff]
        selected = min(available, key=self._selection_key, default=None)
        if selected is None:
            self._commitment = None
            handoff = min((item for item in candidates if item.handoff),
                          key=lambda item: item.prediction.time_until_contact, default=None)
            if handoff is not None:
                return ControllerDecision(
                    "stay", "机器人1已经覆盖这颗小球的接球位置；我继续观察其他来球。",
                    "handoff", handoff.target_x, handoff.prediction.ball_id, handoff.prediction.ball_kind,
                    handoff.contact_side, handoff.ai_distance, handoff.prediction.updates_until_contact,
                    handoff.prediction.time_until_contact, handoff.requires_partner, handoff.prediction.opportunity_id,
                    "none", public, conflict, handoff.player_distance,
                    player_closest.prediction.ball_id if player_closest else None,
                    player_closest.player_distance if player_closest else None,
                )
            return ControllerDecision(
                "stay", "当前没有机器人2能及时承担的来球。", "no_urgent",
                candidates=public, conflict_ball_ids=conflict,
                player_closest_ball_id=player_closest.prediction.ball_id if player_closest else None,
                player_closest_distance_cells=player_closest.player_distance if player_closest else None,
            )

        if selected.requires_partner:
            self._commitment = _Commitment(
                selected.prediction.ball_id, selected.prediction.opportunity_id, selected.contact_side,
            )
        else:
            self._commitment = None
        return self._decision(frame, selected, public, conflict, player_closest,
                              commitment_state="approaching_large" if selected.requires_partner else "none",
                              candidates=candidates)

    def _committed_candidate(self, candidates: list[_Candidate]) -> _Candidate | None:
        if self._commitment is None:
            return None
        committed = next((item for item in candidates if item.requires_partner and (
            item.prediction.ball_id, item.prediction.opportunity_id, item.contact_side,
        ) == (self._commitment.ball_id, self._commitment.opportunity_id, self._commitment.contact_side)), None)
        if committed is None:
            self._commitment = None
            return None
        # Preserve a valid own-side preparation through the actual encounter,
        # even if a tempting small ball appears. Only a proven loss of Robot
        # 2's own reachability releases this particular large-ball plan.
        if not committed.ai_viable:
            self._commitment = None
            return None
        return committed

    def _decision(self, frame: PongFrame, selected: _Candidate,
                  public: tuple[dict[str, Any], ...], conflict: tuple[str, ...],
                  player_closest: _Candidate | None, *, commitment_state: str,
                  candidates: list[_Candidate] | None = None) -> ControllerDecision:
        difference = selected.target_x - self._own_x(frame)
        action = "right" if difference > 0.025 else "left" if difference < -0.025 else "stay"
        if selected.requires_partner:
            intent = "hold_large" if action == "stay" else "catch_large"
            reason = "我已覆盖合作大球分配侧，等待本次接球结算。" if action == "stay" else "我在靠近合作大球的分配接触点。"
        else:
            intent = "hold_target" if action == "stay" else "catch_small"
            reason = "我已覆盖小球接球位置。" if action == "stay" else "我在靠近小球接球位置。"
        small_first_ball_id, small_first_feasible = self._small_first_feasibility(selected, candidates or [])
        return ControllerDecision(
            action, reason, intent, selected.target_x, selected.prediction.ball_id,
            selected.prediction.ball_kind, selected.contact_side, selected.ai_distance,
            selected.prediction.updates_until_contact, selected.prediction.time_until_contact,
            selected.requires_partner, selected.prediction.opportunity_id, commitment_state,
            public, conflict, selected.player_distance,
            player_closest.prediction.ball_id if player_closest else None,
            player_closest.player_distance if player_closest else None,
            small_first_ball_id, small_first_feasible,
        )

    def _candidates(self, frame: PongFrame) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        for ball in frame.balls:
            prediction = predict_next_contact(ball, frame, self.config)
            if prediction is None:
                continue
            candidates.append(self._large_candidate(prediction, frame) if prediction.ball_kind == "large"
                              else self._small_candidate(prediction, frame))
        return candidates

    def _small_candidate(self, prediction: PredictedContact, frame: PongFrame) -> _Candidate:
        cell = prediction.contact_cells[0]
        own_x, other_x = self._own_x(frame), self._other_x(frame)
        target_x = paddle_target_left(own_x, cell, self.config)
        player_target = paddle_target_left(other_x, cell, self.config)
        ai_distance, player_distance = abs(target_x - own_x), abs(player_target - other_x)
        ai_viable = self._travel_seconds(ai_distance) <= prediction.time_until_contact + _EPSILON
        player_covers = paddle_covers_cell(other_x, cell, self.config)
        return _Candidate(
            prediction, target_x, None, False, ai_distance, player_distance,
            ai_viable, ai_viable, player_covers,
            "机器人1已覆盖接球位置" if player_covers else "机器人2可以及时覆盖接球位置",
        )

    def _large_candidate(self, prediction: PredictedContact, frame: PongFrame) -> _Candidate:
        left_cell, right_cell = prediction.contact_cells
        options: list[tuple[float, float, str, float, float]] = []
        for cell, side in ((left_cell, "left"), (right_cell, "right")):
            other = right_cell if side == "left" else left_cell
            own_x, other_x = self._own_x(frame), self._other_x(frame)
            target = paddle_target_left(own_x, cell, self.config)
            player_target = paddle_target_left(other_x, other, self.config)
            ai_distance, player_distance = abs(target - own_x), abs(player_target - other_x)
            options.append((self._travel_seconds(ai_distance) + self._travel_seconds(player_distance),
                            target, side, ai_distance, player_distance))
        _cost, target_x, side, ai_distance, player_distance = min(options, key=lambda option: option[0])
        ai_viable = self._travel_seconds(ai_distance) <= prediction.time_until_contact + _EPSILON
        viable = ai_viable and self._travel_seconds(player_distance) <= prediction.time_until_contact + _EPSILON
        return _Candidate(
            prediction, target_x, side, True, ai_distance, player_distance, viable, ai_viable, False,
            "双方按当前距离可覆盖合作大球两侧" if viable else "当前双方无法同时覆盖合作大球两侧",
        )

    def _selection_key(self, candidate: _Candidate) -> tuple[float, float, float, str]:
        # A large-ball miss costs three, so urgency is compared against that
        # cost. This is a prioritisation rule, not a claim that every large
        # ball is automatically worth abandoning an immediately catchable one.
        slack = max(0.0, candidate.prediction.time_until_contact - self._travel_seconds(candidate.ai_distance))
        return (slack / candidate.penalty, -float(candidate.penalty), candidate.ai_distance, candidate.prediction.ball_id)

    def _small_first_feasibility(self, selected: _Candidate,
                                 candidates: list[_Candidate]) -> tuple[str | None, bool | None]:
        if not selected.requires_partner:
            return None, None
        small = [item for item in candidates if item.prediction.ball_kind == "small" and item.viable and not item.handoff]
        if not small:
            return None, None
        first = min(small, key=self._selection_key)
        # The second leg starts when the small ball reaches and is actually
        # caught at its own contact line, rather than when Robot 2 merely
        # reaches its waiting position.
        reach_small = self._travel_seconds(first.ai_distance)
        leave_small_at = max(reach_small, first.prediction.time_until_contact)
        transfer = self._travel_seconds(abs(selected.target_x - first.target_x))
        player_ready = self._travel_seconds(selected.player_distance) <= selected.prediction.time_until_contact + _EPSILON
        feasible = leave_small_at + transfer <= selected.prediction.time_until_contact + _EPSILON and player_ready
        return first.prediction.ball_id, feasible

    @staticmethod
    def _conflicts(candidates: list[_Candidate]) -> tuple[str, ...]:
        viable = [item for item in candidates if item.viable and not item.handoff]
        if len(viable) < 2:
            return ()
        first, second = sorted(viable, key=lambda item: item.prediction.time_until_contact)[:2]
        if second.prediction.time_until_contact - first.prediction.time_until_contact <= 0.75:
            return first.prediction.ball_id, second.prediction.ball_id
        return ()

    def _travel_seconds(self, distance: float) -> float:
        return max(0.0, distance) / self.config.paddle_speed_per_second
