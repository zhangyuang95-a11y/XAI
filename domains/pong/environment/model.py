from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Any


class BallKind(str, Enum):
    SMALL = "small"
    LARGE = "large"


class EncounterOutcome(str, Enum):
    CAUGHT = "caught"
    MISSED = "missed"


@dataclass
class Ball:
    """A persistent ball. Its stable ID never changes during a run."""

    ball_id: str
    kind: BallKind
    radius: float
    # Continuous top-left positions measured in grid cells. Rendering, wall
    # collision and contact use these same coordinates.
    x: float
    y: float
    vx: float
    vy: float
    # Previous positions are saved for replay and diagnostics, not for a
    # second, independent visual physics system.
    previous_x: int | None = None
    previous_y: int | None = None
    width_cells: int = 1
    height_cells: int = 1
    encounter_index: int = 0
    descending_encounter: bool = False
    last_outcome: str | None = None
    pending_miss: bool = False
    pending_miss_id: str | None = None
    # Curriculum scenarios keep every stable ball slot but may temporarily
    # deactivate a ball. Inactive balls never move, collide or score.
    active: bool = True

    @property
    def is_large(self) -> bool:
        return self.kind is BallKind.LARGE

    @property
    def half_width(self) -> float:
        return self.width_cells / 2

    @property
    def half_height(self) -> float:
        return self.height_cells / 2

    def clone(self) -> "Ball":
        # Snapshots and tests must not accidentally make arbitrary runtime
        # attributes part of authoritative physics state.
        return Ball(**{field.name: getattr(self, field.name) for field in fields(Ball)})

    def to_dict(self) -> dict[str, Any]:
        previous_x = self.x if self.previous_x is None else self.previous_x
        previous_y = self.y if self.previous_y is None else self.previous_y
        return self.__dict__.copy() | {
            "kind": self.kind.value,
            # Grid size determines the occupied rectangle; x/y are allowed to
            # be fractional while an object crosses between visible cells.
            "grid_x": self.x,
            "grid_y": self.y,
            "occupied_cells": {
                "left": self.x,
                "top": self.y,
                "width": self.width_cells,
                "height": self.height_cells,
            },
            "previous_grid_x": previous_x,
            "previous_grid_y": previous_y,
        }


@dataclass(frozen=True)
class PongFrame:
    frame: int
    time_seconds: float
    phase: str
    # Continuous left anchors for four-cell paddles.
    player_x: float
    ai_x: float
    balls: tuple[Ball, ...]
    total_opportunities: int
    successful_opportunities: int
    missed_balls: int
    missed_by_type: dict[str, int]
    encounter_events: tuple[dict[str, Any], ...]
    terminal: bool
    domain_id: str = "pong"
    version: str = "pong-continuous-24x14-v7"

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "time_seconds": self.time_seconds,
            "phase": self.phase,
            "player_x": self.player_x,
            "ai_x": self.ai_x,
            "player_col": self.player_x,
            "ai_col": self.ai_x,
            "balls": [ball.to_dict() for ball in self.balls],
            "total_opportunities": self.total_opportunities,
            "successful_opportunities": self.successful_opportunities,
            "missed_balls": self.missed_balls,
            "missed_by_type": dict(self.missed_by_type),
            "encounter_events": [dict(event) for event in self.encounter_events],
            "terminal": self.terminal,
            "domain_id": self.domain_id,
            "version": self.version,
        }


@dataclass(frozen=True)
class PongTransition:
    before: PongFrame
    after: PongFrame
    player_action: str
    ai_action: str
    events: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "player_action": self.player_action,
            "ai_action": self.ai_action,
            "events": [dict(event) for event in self.events],
        }
