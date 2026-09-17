from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import random
from typing import Any, Mapping

from ..config import PongConfig
from .model import Ball, BallKind, PongFrame, PongTransition


VALID_ACTIONS = ("left", "right", "stay")


@dataclass(frozen=True)
class PredictedContact:
    """The next catch-line crossing computed with the engine's grid rules."""

    ball_id: str
    ball_kind: str
    opportunity_id: str
    updates_until_contact: int
    contact_frame: int
    ball_left: int
    ball_top: int
    contact_cells: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ball_id": self.ball_id,
            "ball_kind": self.ball_kind,
            "opportunity_id": self.opportunity_id,
            "updates_until_contact": self.updates_until_contact,
            "contact_frame": self.contact_frame,
            "ball_left": self.ball_left,
            "ball_top": self.ball_top,
            "contact_cells": list(self.contact_cells),
        }


def ball_contact_cells(ball: Ball, config: PongConfig) -> tuple[int, ...]:
    """Return the exact lower cells used for the next catch-line test."""
    if ball.is_large:
        return (
            ball.x + config.large_contact_left_offset,
            ball.x + config.large_contact_right_offset,
        )
    return (ball.x,)


def paddle_covers_cell(paddle_left: int, cell: int, config: PongConfig) -> bool:
    return paddle_left <= cell < paddle_left + int(config.paddle_width)


def paddle_target_left(paddle_left: int, cell: int, config: PongConfig) -> int:
    """Closest legal left anchor that covers ``cell``."""
    width = int(config.paddle_width)
    if paddle_left <= cell < paddle_left + width:
        return paddle_left
    candidate = cell if cell < paddle_left else cell - width + 1
    return max(0, min(int(config.width) - width, candidate))


def _advance_grid_ball(ball: Ball, config: PongConfig) -> tuple[bool, int, int]:
    """Advance one ball using the authoritative integer-grid motion.

    Lower-edge rebounding is left to the caller, which must first settle an
    encounter at the catch line and record any miss.
    """
    old_x, old_y = ball.x, ball.y
    interval = config.ball_step_interval_updates
    ball.motion_phase = (ball.motion_phase + 1) % interval
    if ball.motion_phase != 0:
        return False, old_x, old_y
    ball.previous_x, ball.previous_y = old_x, old_y
    max_x = int(config.width) - ball.width_cells
    ball.x += int(ball.vx)
    if ball.x < 0:
        ball.x = 0
        ball.vx = abs(int(ball.vx))
    elif ball.x > max_x:
        ball.x = max_x
        ball.vx = -abs(int(ball.vx))
    ball.y += int(ball.vy)
    if ball.y < 0:
        ball.y = 0
        ball.vy = abs(int(ball.vy))
        ball.descending_encounter = False
    return True, old_x, old_y


def predict_next_contact(ball: Ball, frame: PongFrame, config: PongConfig) -> PredictedContact | None:
    """Predict a ball's next downward crossing through the shared motion helper."""
    probe = ball.clone()
    lower_top = int(config.height) - probe.height_cells
    for offset in range(1, max(1, config.max_frames - frame.frame) + 1):
        moved, _old_x, old_y = _advance_grid_ball(probe, config)
        if not moved:
            continue
        old_bottom = old_y + probe.height_cells - 1
        new_bottom = probe.y + probe.height_cells - 1
        if probe.vy > 0 and not probe.descending_encounter and old_bottom < int(config.paddle_y) <= new_bottom:
            return PredictedContact(
                ball_id=probe.ball_id,
                ball_kind=probe.kind.value,
                opportunity_id=f"{probe.ball_id}:{probe.encounter_index + 1}",
                updates_until_contact=offset,
                contact_frame=frame.frame + offset,
                ball_left=probe.x,
                ball_top=probe.y,
                contact_cells=ball_contact_cells(probe, config),
            )
        if probe.y > lower_top:
            probe.y = lower_top
            probe.vy = -abs(int(probe.vy))
            probe.descending_encounter = False
    return None


class PongEnvironment:
    """Fixed-time-step four-ball Pong with one authoritative integer grid."""

    ACTIONS = VALID_ACTIONS

    def __init__(self, config: PongConfig | None = None, *, seed: int | None = None) -> None:
        self.config = config or PongConfig()
        self.config.validate()
        self.seed = self.config.seed if seed is None else int(seed)
        self._rng = random.Random(self.seed)
        self.reset(seed=self.seed)

    def reset(self, seed: int | None = None) -> PongFrame:
        if seed is not None:
            self.seed = int(seed)
            self._rng = random.Random(self.seed)
        self.frame_index = 0
        width = int(self.config.paddle_width)
        self.player_x = max(0, min(int(self.config.width) - width, int(self.config.width * 0.30) - width // 2))
        self.ai_x = max(0, min(int(self.config.width) - width, int(self.config.width * 0.70) - width // 2))
        self.player_motion_phase = self.config.paddle_step_interval_updates - 1
        self.ai_motion_phase = self.config.paddle_step_interval_updates - 1
        self.player_last_action = "stay"
        self.ai_last_action = "stay"
        self.phase = "active"
        self.missed_balls = 0
        self.successful_opportunities = 0
        self.total_opportunities = 0
        self.missed_by_type = {"small": 0, "large": 0}
        self.encounter_events: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []
        self.balls = self._initial_balls()
        return self.frame()

    def _initial_balls(self) -> list[Ball]:
        starts = {"A1": (6, 3), "A2": (22, 7), "A3": (15, 10), "B1": (10, 1), "B2": (20, 8)}
        balls: list[Ball] = []
        for ball_id in self.config.ball_ids:
            base_x, base_y = starts[ball_id]
            large = ball_id in self.config.large_ball_ids
            width = self.config.large_ball_width_cells if large else self.config.small_ball_width_cells
            height = self.config.large_ball_height_cells if large else self.config.small_ball_height_cells
            x = max(0, min(int(self.config.width) - width, base_x + self._rng.choice((-1, 0, 1))))
            balls.append(Ball(
                ball_id=ball_id,
                kind=BallKind.LARGE if large else BallKind.SMALL,
                radius=self.config.large_radius if large else self.config.small_radius,
                x=x, y=base_y, vx=-1 if self._rng.random() < 0.5 else 1, vy=1,
                width_cells=width, height_cells=height,
            ))
        return balls

    @property
    def terminal(self) -> bool:
        return self.phase == "terminal"

    @property
    def time_seconds(self) -> float:
        return self.frame_index * self.config.fixed_dt

    def frame(self) -> PongFrame:
        return PongFrame(
            frame=self.frame_index, time_seconds=self.time_seconds, phase=self.phase,
            player_x=self.player_x, ai_x=self.ai_x, balls=tuple(ball.clone() for ball in self.balls),
            total_opportunities=self.total_opportunities, successful_opportunities=self.successful_opportunities,
            missed_balls=self.missed_balls, missed_by_type=dict(self.missed_by_type),
            encounter_events=tuple(dict(event) for event in self.encounter_events[-12:]),
            terminal=self.terminal, domain_id=self.config.domain_id, version=self.config.version,
        )

    def public_observation(self) -> dict[str, Any]:
        return self.frame().to_dict() | {
            "actions": list(self.ACTIONS), "ball_ids": list(self.config.ball_ids),
            "grid_columns": self.config.grid_columns, "grid_rows": self.config.grid_rows,
            "paddle_y": self.config.paddle_y, "paddle_width_cells": self.config.paddle_width,
            "paddle_height_cells": self.config.paddle_height_cells,
            "fixed_dt": self.config.fixed_dt, "max_frames": self.config.max_frames,
            "ball_step_interval_updates": self.config.ball_step_interval_updates,
        }

    def features(self, agent: str = "ai") -> dict[str, float]:
        own, other = (self.ai_x, self.player_x) if agent == "ai" else (self.player_x, self.ai_x)
        values: dict[str, float] = {
            "self.paddle_x": own / self.config.width,
            "other.paddle_x": other / self.config.width,
            "time.progress": min(1.0, self.frame_index / max(1, self.config.max_frames)),
        }
        for ball in self.balls:
            prefix = f"ball.{ball.ball_id}"
            values.update({
                f"{prefix}.x": ball.x / self.config.width, f"{prefix}.y": ball.y / self.config.height,
                f"{prefix}.vx": float(ball.vx), f"{prefix}.vy": float(ball.vy),
                f"{prefix}.large": float(ball.is_large), f"{prefix}.descending": float(ball.vy > 0),
                f"{prefix}.encounters": float(ball.encounter_index),
            })
        return values

    def step(self, player_action: str = "stay", ai_action: str = "stay") -> PongTransition:
        if player_action not in VALID_ACTIONS or ai_action not in VALID_ACTIONS:
            raise ValueError(f"Actions must be one of {VALID_ACTIONS}")
        before = self.frame()
        if self.terminal:
            return PongTransition(before, before, player_action, ai_action, ())
        self.frame_index += 1
        self.player_x = self._move_paddle(self.player_x, player_action, "player")
        self.ai_x = self._move_paddle(self.ai_x, ai_action, "ai")
        assert 0 <= self.player_x <= int(self.config.width) - int(self.config.paddle_width)
        assert 0 <= self.ai_x <= int(self.config.width) - int(self.config.paddle_width)
        events: list[dict[str, Any]] = []
        for ball in self.balls:
            events.extend(self._advance_ball(ball))
        if self.frame_index >= self.config.max_frames:
            self.phase = "terminal"
        self.encounter_events = events
        if events:
            self.history.extend(deepcopy(events))
        return PongTransition(before, self.frame(), player_action, ai_action, tuple(events))

    def _move_paddle(self, left: int, action: str, paddle: str) -> int:
        phase_name, action_name = f"{paddle}_motion_phase", f"{paddle}_last_action"
        if action == "stay":
            setattr(self, phase_name, self.config.paddle_step_interval_updates - 1)
            setattr(self, action_name, action)
            return left
        if getattr(self, action_name) != action:
            # Direction changes are responsive; holding the direction then
            # advances at the configured cells-per-second rate.
            setattr(self, phase_name, self.config.paddle_step_interval_updates - 1)
        phase = (int(getattr(self, phase_name)) + 1) % self.config.paddle_step_interval_updates
        setattr(self, phase_name, phase)
        setattr(self, action_name, action)
        if phase != 0:
            return left
        delta = -1 if action == "left" else 1 if action == "right" else 0
        return max(0, min(int(self.config.width) - int(self.config.paddle_width), left + delta))

    def _advance_ball(self, ball: Ball) -> list[dict[str, Any]]:
        moved, _old_x, old_y = _advance_grid_ball(ball, self.config)
        if not moved:
            return []
        events: list[dict[str, Any]] = []
        old_bottom = old_y + ball.height_cells - 1
        new_bottom = ball.y + ball.height_cells - 1
        if ball.vy > 0 and not ball.descending_encounter and old_bottom < int(self.config.paddle_y) <= new_bottom:
            event = self._resolve_encounter(ball)
            events.append(event)
            ball.descending_encounter = True
            if event["outcome"] == "caught":
                ball.pending_miss = False
                ball.pending_miss_id = None
                ball.y = int(self.config.paddle_y) - ball.height_cells
                ball.vy = -1
        if ball.vy < 0 and ball.y + ball.height_cells - 1 < int(self.config.paddle_y):
            ball.descending_encounter = False
        lower_top = int(self.config.height) - ball.height_cells
        if ball.y > lower_top:
            ball.y = lower_top
            ball.vy = -1
            ball.descending_encounter = False
            if ball.pending_miss:
                miss_penalty = 3 if ball.is_large else 1
                self.missed_balls += miss_penalty
                kind = ball.kind.value
                self.missed_by_type[kind] += miss_penalty
                events.append({"event": "miss_scored", "encounter_id": ball.pending_miss_id,
                               "ball_id": ball.ball_id, "ball_kind": kind, "frame": self.frame_index,
                               "time_seconds": self.time_seconds, "miss_penalty": miss_penalty})
                ball.pending_miss = False
                ball.pending_miss_id = None
            events.append({"event": "bottom_bounce", "ball_id": ball.ball_id,
                           "frame": self.frame_index, "time_seconds": self.time_seconds})
        return events

    def _resolve_encounter(self, ball: Ball) -> dict[str, Any]:
        contacts = ball_contact_cells(ball, self.config)
        player = tuple(paddle_covers_cell(self.player_x, cell, self.config) for cell in contacts)
        ai = tuple(paddle_covers_cell(self.ai_x, cell, self.config) for cell in contacts)
        if ball.is_large:
            caught = (player[0] and ai[1]) or (player[1] and ai[0])
            coverage: dict[str, Any] = {"contacts": list(contacts), "player": list(player),
                                        "ai": list(ai), "distinct_paddles": caught}
        else:
            caught = player[0] or ai[0]
            coverage = {"contact": contacts[0], "player": player[0], "ai": ai[0]}
        ball.encounter_index += 1
        encounter_id = f"{ball.ball_id}:{ball.encounter_index}"
        outcome = "caught" if caught else "missed"
        ball.last_outcome = outcome
        self.total_opportunities += 1
        if caught:
            self.successful_opportunities += 1
        else:
            ball.pending_miss = True
            ball.pending_miss_id = encounter_id
        return {"event": "encounter", "encounter_id": encounter_id, "ball_id": ball.ball_id,
                "ball_kind": ball.kind.value, "encounter_index": ball.encounter_index,
                "outcome": outcome, "coverage": coverage, "frame": self.frame_index,
                "time_seconds": self.time_seconds, "player_x": self.player_x, "ai_x": self.ai_x,
                "ball_left": ball.x, "ball_top": ball.y, "contact_cells": list(contacts)}

    def snapshot(self) -> dict[str, Any]:
        return {"version": self.config.version, "domain_id": self.config.domain_id, "seed": self.seed,
                "rng": self._rng.getstate(), "frame_index": self.frame_index,
                "player_x": self.player_x, "ai_x": self.ai_x, "phase": self.phase,
                "player_motion_phase": self.player_motion_phase, "ai_motion_phase": self.ai_motion_phase,
                "player_last_action": self.player_last_action, "ai_last_action": self.ai_last_action,
                "balls": [asdict(ball) | {"kind": ball.kind.value} for ball in self.balls],
                "missed_balls": self.missed_balls, "successful_opportunities": self.successful_opportunities,
                "total_opportunities": self.total_opportunities, "missed_by_type": dict(self.missed_by_type),
                "encounter_events": deepcopy(self.encounter_events), "history": deepcopy(self.history)}

    def restore(self, snapshot: Mapping[str, Any]) -> PongFrame:
        if snapshot.get("version", self.config.version) != self.config.version:
            raise ValueError("Pong snapshot version mismatch")
        self.seed = int(snapshot["seed"])
        self._rng = random.Random()
        self._rng.setstate(_tupleize(snapshot["rng"]))
        self.frame_index = int(snapshot["frame_index"])
        self.player_x, self.ai_x = int(snapshot["player_x"]), int(snapshot["ai_x"])
        self.player_motion_phase = int(snapshot.get("player_motion_phase", self.config.paddle_step_interval_updates - 1))
        self.ai_motion_phase = int(snapshot.get("ai_motion_phase", self.config.paddle_step_interval_updates - 1))
        self.player_last_action = str(snapshot.get("player_last_action", "stay"))
        self.ai_last_action = str(snapshot.get("ai_last_action", "stay"))
        self.phase = str(snapshot["phase"])
        self.balls = [Ball(**dict(item, kind=BallKind(item["kind"]))) for item in snapshot["balls"]]
        self.missed_balls = int(snapshot["missed_balls"])
        self.successful_opportunities = int(snapshot["successful_opportunities"])
        self.total_opportunities = int(snapshot["total_opportunities"])
        self.missed_by_type = dict(snapshot["missed_by_type"])
        self.encounter_events, self.history = deepcopy(snapshot["encounter_events"]), deepcopy(snapshot["history"])
        return self.frame()


def _tupleize(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tupleize(item) for item in value)
    return value
