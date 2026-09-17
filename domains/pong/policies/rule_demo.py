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


@dataclass(frozen=True)
class ControllerDecision:
    """One transparent rule-controller decision for a single fixed update."""

    action: str
    reason: str
    intent_type: str = "no_urgent"
    target_x: int | None = None
    target_ball_id: str | None = None
    target_kind: str | None = None
    contact_side: str | None = None
    distance_cells: int | None = None
    eta_updates: int | None = None
    requires_partner: bool = False
    opportunity_id: str | None = None
    candidates: tuple[dict[str, Any], ...] = ()
    conflict_ball_ids: tuple[str, ...] = ()
    # These are factual comparisons computed from the same prediction used to
    # select the target.  They support an explanation of *why this ball now*,
    # rather than exposing an internal controller label.
    player_distance_cells: int | None = None
    small_first_ball_id: str | None = None
    small_first_feasible: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy() | {"candidates": [dict(item) for item in self.candidates]}


@dataclass(frozen=True)
class _Candidate:
    prediction: PredictedContact
    target_x: int
    contact_side: str | None
    requires_partner: bool
    ai_distance: int
    player_distance: int
    viable: bool
    handoff: bool
    reason: str

    def public(self) -> dict[str, Any]:
        return {
            "ball_id": self.prediction.ball_id,
            "kind": self.prediction.ball_kind,
            "opportunity_id": self.prediction.opportunity_id,
            "eta_updates": self.prediction.updates_until_contact,
            "target_x": self.target_x,
            "contact_side": self.contact_side,
            "ai_distance": self.ai_distance,
            "player_distance": self.player_distance,
            "viable": self.viable,
            "handoff": self.handoff,
            "reason": self.reason,
        }


class RuleDemoController:
    """Rule baseline that allocates all four balls without reading user input.

    It is deliberately labelled as a rule controller.  It shares the engine's
    integer trajectory predictor; it is not a claim that a neural policy made
    the choice.
    """

    version = "rule-demo-grid-three-small-two-large-v6-smooth"

    def __init__(self, config: PongConfig | None = None) -> None:
        self.config = config or PongConfig()
        self._commitment: tuple[str, str, int, str | None] | None = None

    def reset(self) -> None:
        self._commitment = None

    def choose(self, frame: PongFrame) -> ControllerDecision:
        if frame.terminal:
            return ControllerDecision("stay", "本局已经结束。")
        candidates = self._candidates(frame)
        public = tuple(candidate.public() for candidate in candidates)
        viable = [candidate for candidate in candidates if candidate.viable and not candidate.handoff]
        handoffs = [candidate for candidate in candidates if candidate.handoff]
        selected = self._retain_or_select(viable)
        conflict = self._conflicts(viable)
        if selected is None:
            self._commitment = None
            if handoffs:
                target = min(handoffs, key=lambda item: item.prediction.updates_until_contact)
                return ControllerDecision(
                    "stay", "机器人1已经覆盖这颗球的预计接球格；我继续检查其余来球。",
                    "handoff", target.target_x, target.prediction.ball_id, target.prediction.ball_kind,
                    target.contact_side, target.ai_distance, target.prediction.updates_until_contact,
                    target.requires_partner, target.prediction.opportunity_id, public, conflict,
                    target.player_distance,
                )
            return ControllerDecision("stay", "当前没有机器人2能及时承担的来球。", "no_urgent",
                                      candidates=public, conflict_ball_ids=conflict)

        self._commitment = (
            selected.prediction.ball_id, selected.prediction.opportunity_id,
            selected.target_x, selected.contact_side,
        )
        action = "stay"
        if frame.ai_x < selected.target_x:
            action = "right"
        elif frame.ai_x > selected.target_x:
            action = "left"
        if action == "stay":
            reason = "我已覆盖预计接球格，保留位置等待来球。"
            intent = "hold_target"
        elif selected.requires_partner:
            reason = "我在移动到合作大球的分配接触点。"
            intent = "catch_large"
        else:
            reason = "我在移动到小球的预计接球格。"
            intent = "catch_small"
        small_first_ball_id, small_first_feasible = self._small_first_feasibility(selected, candidates)
        return ControllerDecision(
            action, reason, intent, selected.target_x, selected.prediction.ball_id,
            selected.prediction.ball_kind, selected.contact_side, selected.ai_distance,
            selected.prediction.updates_until_contact, selected.requires_partner,
            selected.prediction.opportunity_id, public, conflict, selected.player_distance,
            small_first_ball_id, small_first_feasible,
        )

    def _candidates(self, frame: PongFrame) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        for ball in frame.balls:
            prediction = predict_next_contact(ball, frame, self.config)
            if prediction is None:
                continue
            if prediction.ball_kind == "large":
                candidates.append(self._large_candidate(prediction, frame))
            else:
                candidates.append(self._small_candidate(prediction, frame))
        return candidates

    def _small_candidate(self, prediction: PredictedContact, frame: PongFrame) -> _Candidate:
        cell = prediction.contact_cells[0]
        target_x = paddle_target_left(frame.ai_x, cell, self.config)
        ai_distance = abs(target_x - frame.ai_x)
        player_target = paddle_target_left(frame.player_x, cell, self.config)
        player_distance = abs(player_target - frame.player_x)
        player_covers = paddle_covers_cell(frame.player_x, cell, self.config)
        viable = self._travel_updates(ai_distance) <= prediction.updates_until_contact
        # A player already covering the predicted contact is a hand-off, but it
        # never stops the controller from evaluating the other three balls.
        return _Candidate(
            prediction, target_x, None, False, ai_distance, player_distance,
            viable, player_covers,
            "机器人1已覆盖预计接球格" if player_covers else "机器人2可及时覆盖预计接球格",
        )

    def _large_candidate(self, prediction: PredictedContact, frame: PongFrame) -> _Candidate:
        left_cell, right_cell = prediction.contact_cells
        options: list[tuple[int, int, str, int, int]] = []
        for cell, side in ((left_cell, "left"), (right_cell, "right")):
            other = right_cell if side == "left" else left_cell
            target = paddle_target_left(frame.ai_x, cell, self.config)
            player_target = paddle_target_left(frame.player_x, other, self.config)
            ai_distance, player_distance = abs(target - frame.ai_x), abs(player_target - frame.player_x)
            options.append((ai_distance + player_distance, target, side, ai_distance, player_distance))
        _cost, target_x, side, ai_distance, player_distance = min(options, key=lambda option: option[0])
        viable = (
            self._travel_updates(ai_distance) <= prediction.updates_until_contact
            and self._travel_updates(player_distance) <= prediction.updates_until_contact
        )
        return _Candidate(
            prediction, target_x, side, True, ai_distance, player_distance, viable, False,
            "双方可分别覆盖大球的两个实际接触格" if viable else "双方距离不足以共同覆盖大球接触格",
        )

    def _retain_or_select(self, viable: list[_Candidate]) -> _Candidate | None:
        if not viable:
            return None
        if self._commitment:
            ball_id, opportunity_id, target_x, side = self._commitment
            committed = next((candidate for candidate in viable if (
                candidate.prediction.ball_id, candidate.prediction.opportunity_id,
                candidate.target_x, candidate.contact_side,
            ) == (ball_id, opportunity_id, target_x, side)), None)
            if committed is not None:
                earliest = min(viable, key=lambda item: (item.prediction.updates_until_contact, item.ai_distance))
                if earliest.prediction.updates_until_contact + 2 >= committed.prediction.updates_until_contact:
                    return committed
        return min(viable, key=lambda item: (
            item.prediction.updates_until_contact, item.ai_distance, item.player_distance,
        ))

    def _small_first_feasibility(
        self, selected: _Candidate, candidates: list[_Candidate],
    ) -> tuple[str | None, bool | None]:
        """Report whether an urgent small-ball detour still preserves a large catch.

        This is deliberately an explanatory counterfactual, not an instruction
        sent to either paddle.  It uses only the same current-frame positions
        and fixed-grid travel estimate available to the controller.
        """
        if not selected.requires_partner:
            return None, None
        small = [
            candidate for candidate in candidates
            if candidate.prediction.ball_kind == "small" and candidate.viable and not candidate.handoff
        ]
        if not small:
            return None, None
        first = min(small, key=lambda item: (
            item.prediction.updates_until_contact, item.ai_distance,
        ))
        # Approximate the second leg from the small-ball contact position to
        # this robot's allocated large-ball contact.  The answer is presented
        # as an estimate because both balls continue moving after this frame.
        detour_updates = self._travel_updates(first.ai_distance) + self._travel_updates(
            abs(selected.target_x - first.target_x)
        )
        return first.prediction.ball_id, detour_updates <= selected.prediction.updates_until_contact

    @staticmethod
    def _conflicts(viable: list[_Candidate]) -> tuple[str, ...]:
        if len(viable) < 2:
            return ()
        first, second = sorted(viable, key=lambda item: item.prediction.updates_until_contact)[:2]
        if second.prediction.updates_until_contact - first.prediction.updates_until_contact <= 2:
            return (first.prediction.ball_id, second.prediction.ball_id)
        return ()

    def _travel_updates(self, distance: int) -> int:
        if distance <= 0:
            return 0
        # A newly chosen direction moves immediately, then each remaining cell
        # follows the configured paddle update interval.
        return 1 + (distance - 1) * self.config.paddle_step_interval_updates
