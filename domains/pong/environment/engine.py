from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import random
from typing import Any, Mapping

from ..config import PongConfig
from .model import Ball, BallKind, PongFrame, PongTransition


VALID_ACTIONS = ("left", "right", "stay")
_EPSILON = 1e-6


class PredictedContact:
    """One next downward crossing produced by the same continuous physics."""

    def __init__(self, *, ball_id: str, ball_kind: str, opportunity_id: str,
                 updates_until_contact: int, time_until_contact: float,
                 contact_frame: int, ball_left: float, ball_top: float,
                 contact_cells: tuple[float, ...]) -> None:
        self.ball_id = ball_id
        self.ball_kind = ball_kind
        self.opportunity_id = opportunity_id
        self.updates_until_contact = updates_until_contact
        self.time_until_contact = time_until_contact
        self.contact_frame = contact_frame
        self.ball_left = ball_left
        self.ball_top = ball_top
        self.contact_cells = contact_cells

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy() | {"contact_cells": list(self.contact_cells)}


def ball_contact_cells(ball: Ball, config: PongConfig) -> tuple[float, ...]:
    """Return the actual lower-edge contact coordinates for the ball."""
    if ball.is_large:
        return (
            ball.x + float(config.large_contact_left_offset),
            ball.x + float(config.large_contact_right_offset),
        )
    return (ball.x,)


def paddle_covers_cell(paddle_left: float, cell: float, config: PongConfig) -> bool:
    return paddle_left - _EPSILON <= cell < paddle_left + float(config.paddle_width) - _EPSILON


def paddle_target_left(paddle_left: float, cell: float, config: PongConfig) -> float:
    """Closest legal continuous anchor whose four-cell paddle covers ``cell``."""
    width = float(config.paddle_width)
    maximum = float(config.width) - width
    if paddle_covers_cell(paddle_left, cell, config):
        return max(0.0, min(maximum, paddle_left))
    target = cell if cell < paddle_left else cell - width + _EPSILON
    return max(0.0, min(maximum, target))


def _reflect(position: float, velocity: float, lower: float, upper: float) -> tuple[float, float]:
    """Reflect a continuous coordinate within a closed interval."""
    while position < lower - _EPSILON or position > upper + _EPSILON:
        if position < lower:
            position = lower + (lower - position)
            velocity = abs(velocity)
        elif position > upper:
            position = upper - (position - upper)
            velocity = -abs(velocity)
    return max(lower, min(upper, position)), velocity


def _advance_probe(ball: Ball, config: PongConfig, dt: float) -> tuple[float, float]:
    """Advance a ball without resolving paddle contact; shared by prediction."""
    old_x, old_y = ball.x, ball.y
    ball.x, ball.vx = _reflect(
        ball.x + ball.vx * dt, ball.vx, 0.0, float(config.width - ball.width_cells),
    )
    ball.y, ball.vy = _reflect(
        ball.y + ball.vy * dt, ball.vy, 0.0, float(config.height - ball.height_cells),
    )
    return old_x, old_y


def predict_next_contact(ball: Ball, frame: PongFrame, config: PongConfig) -> PredictedContact | None:
    """Predict the next descending catch-line crossing using production motion."""
    probe = ball.clone()
    elapsed = 0.0
    max_updates = max(1, config.max_frames - frame.frame)
    for update in range(1, max_updates + 1):
        old_x, old_y = _advance_probe(probe, config, config.fixed_dt)
        elapsed += config.fixed_dt
        old_bottom = old_y + probe.height_cells
        new_bottom = probe.y + probe.height_cells
        if probe.vy > 0 and not probe.descending_encounter and old_bottom < config.paddle_y <= new_bottom:
            fraction = (config.paddle_y - old_bottom) / max(_EPSILON, new_bottom - old_bottom)
            contact_x = old_x + (probe.x - old_x) * fraction
            contact_cells = (
                (contact_x + config.large_contact_left_offset, contact_x + config.large_contact_right_offset)
                if probe.is_large else (contact_x,)
            )
            return PredictedContact(
                ball_id=probe.ball_id, ball_kind=probe.kind.value,
                opportunity_id=f"{probe.ball_id}:{probe.encounter_index + 1}",
                updates_until_contact=update,
                time_until_contact=max(0.0, elapsed - config.fixed_dt + fraction * config.fixed_dt),
                contact_frame=frame.frame + update, ball_left=contact_x,
                ball_top=config.paddle_y - probe.height_cells, contact_cells=contact_cells,
            )
        if probe.vy > 0 and probe.y >= float(config.height - probe.height_cells) - _EPSILON:
            probe.descending_encounter = False
    return None


class PongEnvironment:
    """Authoritative 60 Hz continuous physics on a grid-sized court."""

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
        width = float(self.config.paddle_width)
        self.player_x = max(0.0, min(self.config.width - width, self.config.width * 0.28 - width / 2))
        self.ai_x = max(0.0, min(self.config.width - width, self.config.width * 0.72 - width / 2))
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
        starts = {"A1": (3.0, 2.0), "A2": (18.0, 5.0), "A3": (12.0, 8.0),
                  "B1": (7.0, 1.0), "B2": (16.0, 6.0)}
        balls: list[Ball] = []
        for ball_id in self.config.ball_ids:
            base_x, base_y = starts[ball_id]
            large = ball_id in self.config.large_ball_ids
            width = self.config.large_ball_width_cells if large else self.config.small_ball_width_cells
            height = self.config.large_ball_height_cells if large else self.config.small_ball_height_cells
            x = max(0.0, min(float(self.config.width - width), base_x + self._rng.choice((-0.35, 0.0, 0.35))))
            balls.append(Ball(
                ball_id=ball_id, kind=BallKind.LARGE if large else BallKind.SMALL,
                radius=self.config.large_radius if large else self.config.small_radius,
                x=x, y=base_y,
                vx=(-self.config.ball_speed_x_per_second if self._rng.random() < 0.5 else self.config.ball_speed_x_per_second),
                vy=self.config.ball_speed_y_per_second, width_cells=width, height_cells=height,
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
            "continuous_motion": True,
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
                f"{prefix}.vx": ball.vx / self.config.ball_speed_x_per_second,
                f"{prefix}.vy": ball.vy / self.config.ball_speed_y_per_second,
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
        self.player_x = self._move_paddle(self.player_x, player_action)
        self.ai_x = self._move_paddle(self.ai_x, ai_action)
        self.player_last_action, self.ai_last_action = player_action, ai_action
        events: list[dict[str, Any]] = []
        for ball in self.balls:
            events.extend(self._advance_ball(ball))
        if self.frame_index >= self.config.max_frames:
            self.phase = "terminal"
        self.encounter_events = events
        if events:
            self.history.extend(deepcopy(events))
        return PongTransition(before, self.frame(), player_action, ai_action, tuple(events))

    def _move_paddle(self, left: float, action: str) -> float:
        delta = -1.0 if action == "left" else 1.0 if action == "right" else 0.0
        maximum = self.config.width - self.config.paddle_width
        return max(0.0, min(maximum, left + delta * self.config.paddle_speed_per_second * self.config.fixed_dt))

    def _advance_ball(self, ball: Ball) -> list[dict[str, Any]]:
        old_x, old_y = ball.x, ball.y
        ball.previous_x, ball.previous_y = old_x, old_y
        ball.x, ball.vx = _reflect(
            old_x + ball.vx * self.config.fixed_dt, ball.vx, 0.0, float(self.config.width - ball.width_cells),
        )
        if ball.vy < 0:
            ball.y, ball.vy = _reflect(
                old_y + ball.vy * self.config.fixed_dt, ball.vy, 0.0,
                float(self.config.height - ball.height_cells),
            )
        else:
            # A downward ball is allowed to reach the lower edge so this
            # method can score a missed catch exactly once before bouncing.
            ball.y = old_y + ball.vy * self.config.fixed_dt
        events: list[dict[str, Any]] = []
        old_bottom, new_bottom = old_y + ball.height_cells, ball.y + ball.height_cells
        if ball.vy > 0 and not ball.descending_encounter and old_bottom < self.config.paddle_y <= new_bottom:
            fraction = (self.config.paddle_y - old_bottom) / max(_EPSILON, new_bottom - old_bottom)
            contact_x = old_x + (ball.x - old_x) * fraction
            event = self._resolve_encounter(ball, contact_x)
            events.append(event)
            ball.descending_encounter = True
            if event["outcome"] == "caught":
                ball.pending_miss = False
                ball.pending_miss_id = None
                ball.y = self.config.paddle_y - ball.height_cells
                ball.vy = -abs(ball.vy)
        lower_top = float(self.config.height - ball.height_cells)
        if ball.vy > 0 and ball.y >= lower_top - _EPSILON:
            ball.y = lower_top
            ball.vy = -abs(ball.vy)
            ball.descending_encounter = False
            if ball.pending_miss:
                penalty = 3 if ball.is_large else 1
                self.missed_balls += penalty
                self.missed_by_type[ball.kind.value] += penalty
                events.append({"event": "miss_scored", "encounter_id": ball.pending_miss_id,
                               "ball_id": ball.ball_id, "ball_kind": ball.kind.value,
                               "frame": self.frame_index, "time_seconds": self.time_seconds,
                               "miss_penalty": penalty})
                ball.pending_miss = False
                ball.pending_miss_id = None
            events.append({"event": "bottom_bounce", "ball_id": ball.ball_id,
                           "frame": self.frame_index, "time_seconds": self.time_seconds})
        if ball.vy < 0 and ball.y + ball.height_cells < self.config.paddle_y - _EPSILON:
            ball.descending_encounter = False
        return events

    def _resolve_encounter(self, ball: Ball, contact_left: float) -> dict[str, Any]:
        contacts = ((contact_left, contact_left + self.config.large_contact_right_offset)
                    if ball.is_large else (contact_left,))
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
                "ball_left": contact_left, "ball_top": self.config.paddle_y - ball.height_cells,
                "contact_cells": list(contacts)}

    def snapshot(self) -> dict[str, Any]:
        return {"version": self.config.version, "domain_id": self.config.domain_id, "seed": self.seed,
                "rng": self._rng.getstate(), "frame_index": self.frame_index,
                "player_x": self.player_x, "ai_x": self.ai_x, "phase": self.phase,
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
        self.player_x, self.ai_x = float(snapshot["player_x"]), float(snapshot["ai_x"])
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
