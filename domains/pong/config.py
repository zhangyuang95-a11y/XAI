from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PongConfig:
    """Versioned grid/continuous four-ball physics configuration."""

    version: str = "pong-grid-three-small-two-large-v6-smooth"
    domain_id: str = "pong"
    # Physics coordinates are grid columns/rows. Rendering uses the same
    # values, so a displayed cell and a collision cell cannot diverge.
    width: float = 30.0
    height: float = 18.0
    # Row 15 is the catch line; rows 16 and 17 stay visibly empty below it.
    paddle_y: float = 15.0
    paddle_width: float = 4.0
    paddle_height_cells: int = 1
    paddle_speed_per_second: float = 5.0
    fixed_hz: int = 20
    duration_seconds: float = 90.0
    small_radius: float = 0.5
    large_radius: float = 1.5
    # 2.5 cells per second at 20 Hz gives one integer-grid move every eight
    # updates. The browser interpolates these authoritative cell changes.
    # Positions are integer top-left cell anchors; movement phase is saved.
    ball_speed_y_per_second: float = 2.5
    ball_speed_x_per_second: float = 2.5
    small_ball_width_cells: int = 1
    small_ball_height_cells: int = 1
    # A large ball occupies twice the side length of a small ball, not a
    # rectangular obstacle. Its two lower cells remain the cooperation ports.
    large_ball_width_cells: int = 2
    large_ball_height_cells: int = 2
    large_contact_left_offset: int = 0
    large_contact_right_offset: int = 1
    ball_ids: tuple[str, ...] = ("A1", "A2", "A3", "B1", "B2")
    small_ball_ids: tuple[str, ...] = ("A1", "A2", "A3")
    large_ball_ids: tuple[str, ...] = ("B1", "B2")
    player_actions: tuple[str, ...] = ("left", "right", "stay")
    control_mode: str = "rule_demo"
    study_protocol: str = "coordination_explanation"
    seed: int = 260918

    @property
    def fixed_dt(self) -> float:
        return 1.0 / self.fixed_hz

    @property
    def max_frames(self) -> int:
        return int(round(self.duration_seconds * self.fixed_hz))

    @property
    def paddle_width_percent(self) -> float:
        """Horizontal paddle width in the same 0-100 coordinate system as x."""
        return 100.0 * self.paddle_width / self.width

    @property
    def grid_columns(self) -> int:
        return int(round(self.width))

    @property
    def grid_rows(self) -> int:
        return int(round(self.height))

    @property
    def logic_hz(self) -> int:
        return self.fixed_hz

    @property
    def cell_dt(self) -> float:
        return self.fixed_dt * self.ball_speed_y_per_second

    @property
    def ball_step_interval_updates(self) -> int:
        """Number of logic updates between one grid-cell ball move."""
        if self.ball_speed_y_per_second <= 0:
            raise ValueError("ball_speed_y_per_second must be positive")
        return int(round(self.fixed_hz / self.ball_speed_y_per_second))

    @property
    def paddle_step_interval_updates(self) -> int:
        if self.paddle_speed_per_second <= 0:
            raise ValueError("paddle_speed_per_second must be positive")
        return int(round(self.fixed_hz / self.paddle_speed_per_second))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "fixed_dt": self.fixed_dt,
            "max_frames": self.max_frames,
            "paddle_width_percent": self.paddle_width_percent,
            "grid_columns": self.grid_columns,
            "grid_rows": self.grid_rows,
            "logic_hz": self.logic_hz,
            "cell_dt": self.cell_dt,
            "ball_step_interval_updates": self.ball_step_interval_updates,
            "paddle_step_interval_updates": self.paddle_step_interval_updates,
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PongConfig":
        values = dict(values)
        values.pop("fixed_dt", None)
        values.pop("max_frames", None)
        values.pop("ball_step_interval_updates", None)
        values.pop("paddle_step_interval_updates", None)
        for key in ("ball_ids", "small_ball_ids", "large_ball_ids", "player_actions"):
            if isinstance(values.get(key), list):
                values[key] = tuple(values[key])
        fields = set(cls.__dataclass_fields__)
        unknown = set(values) - fields
        if unknown:
            raise ValueError(f"Unknown Pong config fields: {sorted(unknown)}")
        return cls(**{key: values[key] for key in fields if key in values})

    @classmethod
    def from_json(cls, path: str | Path) -> "PongConfig":
        with Path(path).open(encoding="utf-8") as handle:
            values = json.load(handle)
        if not isinstance(values, dict):
            raise ValueError("Pong config JSON must contain an object")
        return cls.from_mapping(values)

    def validate(self) -> None:
        if self.fixed_hz <= 0 or self.duration_seconds <= 0:
            raise ValueError("fixed_hz and duration_seconds must be positive")
        if self.grid_columns <= 0 or self.grid_rows <= 0:
            raise ValueError("grid width and height must be positive")
        if abs(self.width - self.grid_columns) > 1e-6 or abs(self.height - self.grid_rows) > 1e-6:
            raise ValueError("width and height must be whole grid dimensions")
        if tuple(self.ball_ids) != tuple(self.small_ball_ids + self.large_ball_ids):
            raise ValueError("ball_ids must list three small balls followed by two large balls")
        if tuple(self.small_ball_ids) != ("A1", "A2", "A3") or tuple(self.large_ball_ids) != ("B1", "B2"):
            raise ValueError("the continuous protocol requires A1-A3 and B1-B2")
        if self.paddle_width <= 0 or self.paddle_width >= self.width:
            raise ValueError("paddle_width must be within the board")
        if abs(self.paddle_width - round(self.paddle_width)) > 1e-6:
            raise ValueError("paddle_width must be a whole number of cells")
        if abs(self.paddle_y - round(self.paddle_y)) > 1e-6:
            raise ValueError("paddle_y must be a whole grid row")
        if self.paddle_height_cells != 1:
            raise ValueError("paddles must occupy exactly one grid row")
        if not (0 < self.paddle_y < self.height):
            raise ValueError("paddle_y must be inside the board")
        y_interval = self.fixed_hz / self.ball_speed_y_per_second
        paddle_interval = self.fixed_hz / self.paddle_speed_per_second
        if abs(y_interval - round(y_interval)) > 1e-6 or abs(paddle_interval - round(paddle_interval)) > 1e-6:
            raise ValueError("ball and paddle speeds must produce integral grid movement intervals")
        if self.small_ball_width_cells != 1 or self.small_ball_height_cells != 1:
            raise ValueError("small balls must occupy one cell")
        if self.large_ball_width_cells != 2 or self.large_ball_height_cells != 2:
            raise ValueError("large balls must occupy a 2x2 cell footprint")
        if (self.large_contact_left_offset, self.large_contact_right_offset) != (0, 1):
            raise ValueError("large-ball contacts must be the two bottom corner cells")
        if self.control_mode not in {"rule_demo", "frozen_nn"}:
            raise ValueError("control_mode must be rule_demo or frozen_nn")
        if self.study_protocol not in {"coordination_explanation", "rule_discovery"}:
            raise ValueError("unknown study_protocol")
