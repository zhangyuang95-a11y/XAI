from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import math
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
    """Predict the next descending catch-line crossing using production motion.

    Horizontal and vertical reflections have a closed form. Using it here is
    mathematically the same triangular-wave trajectory as advance_ball, while
    avoiding a 60 Hz probe loop inside every Actor observation.
    """
    if not ball.active:
        return None
    speed_y = abs(ball.vy)
    if speed_y <= _EPSILON:
        return None
    target_top = float(config.paddle_y - ball.height_cells)
    lower, upper = 0.0, float(config.height - ball.height_cells)
    y = max(lower, min(upper, ball.y))
    # The current encounter is available only before the ball has crossed the
    # catch line. Once it was settled, skip to the next legal downward pass.
    if not ball.descending_encounter and ball.vy > 0 and y <= target_top + _EPSILON:
        elapsed = max(0.0, target_top - y) / speed_y
    else:
        if ball.vy > 0:
            elapsed = max(0.0, upper - y) / speed_y
            y = upper
        else:
            elapsed = max(0.0, y - lower) / speed_y
            y = lower
        # From the lower wall the ball rises to the top, then falls to the
        # next catch line. This also handles an already-settled miss.
        if y > lower + _EPSILON:
            elapsed += (y - lower) / speed_y
        elapsed += target_top / speed_y
    span = float(config.width - ball.width_cells)
    if span <= _EPSILON or abs(ball.vx) <= _EPSILON:
        contact_x = 0.0 if span <= _EPSILON else ball.x
    else:
        phase = ball.x if ball.vx > 0 else 2.0 * span - ball.x
        wrapped = (phase + abs(ball.vx) * elapsed) % (2.0 * span)
        contact_x = wrapped if wrapped <= span else 2.0 * span - wrapped
    contact_cells = (
        (contact_x + config.large_contact_left_offset, contact_x + config.large_contact_right_offset)
        if ball.is_large else (contact_x,)
    )
    updates = max(1, int(math.ceil(elapsed / config.fixed_dt - _EPSILON)))
    if frame.frame + updates > config.max_frames:
        return None
    return PredictedContact(
        ball_id=ball.ball_id, ball_kind=ball.kind.value,
        opportunity_id=f"{ball.ball_id}:{ball.encounter_index + 1}",
        updates_until_contact=updates, time_until_contact=elapsed,
        contact_frame=frame.frame + updates, ball_left=contact_x,
        ball_top=target_top, contact_cells=contact_cells,
    )


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
        self.training_scenario = "full"
        self.training_scenario_metadata: dict[str, Any] = {"scenario": "full", "source": "normal_reset"}
        return self.frame()

    def reset_training_episode(self, *, seed: int, scenario_name: str,
                               active_ball_ids: set[str]) -> PongFrame:
        """Reset a legal curriculum episode without changing the ball slots.

        The five stable IDs remain in observations.  An inactive slot is
        explicit and inert, rather than being hidden by an out-of-board value.
        """
        self.reset(seed=seed)
        unknown = set(active_ball_ids) - set(self.config.ball_ids)
        if unknown:
            raise ValueError(f"unknown curriculum ball IDs: {sorted(unknown)}")
        self.training_scenario = str(scenario_name)
        self.training_scenario_metadata: dict[str, Any] = {
            "scenario": str(scenario_name), "source": "normal_reset" if scenario_name == "full" else "physical_curriculum",
        }
        # The consolidation stage is the real game reset: all five balls,
        # normal start locations and normal 90-second physics. Do not turn it
        # into another hand-placed training state.
        if scenario_name == "full" and active_ball_ids == set(self.config.ball_ids):
            return self.frame()
        for ball in self.balls:
            ball.active = ball.ball_id in active_ball_ids
            if not ball.active:
                ball.vx = ball.vy = 0.0
                ball.pending_miss = False
                ball.pending_miss_id = None
                continue
            # Randomize only legal public physical state. The curriculum never
            # supplies a target action or a hidden target-ball feature.
            maximum_x = float(self.config.width - ball.width_cells)
            ball.x = self._rng.uniform(0.0, maximum_x)
            ball.y = self._rng.uniform(0.5, max(0.6, self.config.paddle_y - ball.height_cells - 1.0))
            ball.vx = self.config.ball_speed_x_per_second * self._rng.choice((-1.0, 1.0))
            ball.vy = self.config.ball_speed_y_per_second
            ball.descending_encounter = False
            ball.encounter_index = 0
            ball.last_outcome = None
        width = float(self.config.paddle_width)
        self.player_x = self._rng.uniform(0.0, float(self.config.width) - width)
        self.ai_x = self._rng.uniform(0.0, float(self.config.width) - width)
        self._place_curriculum_scenario(str(scenario_name))
        return self.frame()

    def _active_ball(self, kind: BallKind | None = None) -> Ball:
        candidates = [ball for ball in self.balls if ball.active and (kind is None or ball.kind is kind)]
        if not candidates:
            raise ValueError(f"curriculum scenario {self.training_scenario} has no requested active ball")
        return self._rng.choice(candidates)

    def _put_descending_near_contact(self, ball: Ball, *, seconds: float) -> PredictedContact:
        """Set one legal, freely moving ball close to its next real contact."""
        seconds = max(.40, min(1.20, float(seconds)))
        ball.vy = abs(self.config.ball_speed_y_per_second)
        ball.y = max(0.0, self.config.paddle_y - ball.height_cells - ball.vy * seconds)
        ball.descending_encounter = False
        ball.pending_miss = False
        ball.pending_miss_id = None
        prediction = predict_next_contact(ball, self.frame(), self.config)
        if prediction is None:
            raise RuntimeError("could not construct legal Pong contact scenario")
        return prediction

    def _place_pair_for_large(self, ball: Ball, *, mode: str) -> None:
        prediction = self._put_descending_near_contact(ball, seconds=self._rng.uniform(.40, 1.20))
        left, right = prediction.contact_cells
        player_left = paddle_target_left(self.player_x, left, self.config)
        ai_right = paddle_target_left(self.ai_x, right, self.config)
        if self._rng.random() < .5:
            player_left, ai_right = paddle_target_left(self.player_x, right, self.config), paddle_target_left(self.ai_x, left, self.config)
        if mode == "ready":
            self.player_x, self.ai_x = player_left, ai_right
        elif mode == "approach":
            travel = min(prediction.time_until_contact * self.config.paddle_speed_per_second * .65, 3.0)
            self.player_x = max(0.0, min(self.config.width - self.config.paddle_width,
                                         player_left + self._rng.choice((-1.0, 1.0)) * travel))
            self.ai_x = max(0.0, min(self.config.width - self.config.paddle_width,
                                     ai_right + self._rng.choice((-1.0, 1.0)) * travel))
        elif mode == "one_holds":
            holder_player = self._rng.random() < .5
            if holder_player:
                self.player_x = player_left
                self.ai_x = max(0.0, min(self.config.width - self.config.paddle_width,
                                         ai_right + self._rng.choice((-1.0, 1.0)) * min(2.0, prediction.time_until_contact * self.config.paddle_speed_per_second * .55)))
            else:
                self.ai_x = ai_right
                self.player_x = max(0.0, min(self.config.width - self.config.paddle_width,
                                             player_left + self._rng.choice((-1.0, 1.0)) * min(2.0, prediction.time_until_contact * self.config.paddle_speed_per_second * .55)))
        else:
            raise ValueError(f"unknown large placement mode {mode}")
        self.training_scenario_metadata.update({
            "ball_id": ball.ball_id, "opportunity_id": prediction.opportunity_id,
            "time_to_contact": prediction.time_until_contact, "mode": mode,
        })

    def _place_curriculum_scenario(self, scenario: str) -> None:
        """Construct legal public state only; physics remains authoritative."""
        if scenario in {"single_small", "single_large", "two_small", "small_large_conflict", "hold_large"}:
            # Legacy v2 names remain loadable for old checkpoint/replay tests.
            # They are intentionally not part of the v2.1 curriculum schedule.
            return
        if scenario == "large_both_ready":
            self._place_pair_for_large(self._active_ball(BallKind.LARGE), mode="ready")
        elif scenario == "large_both_approach":
            self._place_pair_for_large(self._active_ball(BallKind.LARGE), mode="approach")
        elif scenario == "large_one_holds":
            self._place_pair_for_large(self._active_ball(BallKind.LARGE), mode="one_holds")
        elif scenario == "small_then_large":
            small = self._active_ball(BallKind.SMALL)
            large = self._active_ball(BallKind.LARGE)
            # Make both feasible and deliberately competing variants. The
            # distinction is checked from the *actual* small contact followed
            # by the transfer time, never from a visual near-ball heuristic.
            feasible_variant = self._rng.random() < .5
            small.x = self._rng.uniform(3.0, 7.0)
            large.x = (small.x + self._rng.uniform(-.8, .8)) if feasible_variant else (18.0 if small.x < 10.0 else 1.0)
            large.x = max(0.0, min(float(self.config.width - large.width_cells), large.x))
            small_prediction = self._put_descending_near_contact(small, seconds=self._rng.uniform(.40, .65))
            large_prediction = self._put_descending_near_contact(large, seconds=self._rng.uniform(.85, 1.20))
            # Put one paddle on the small ball and the other on one required
            # large contact. The later timing then determines whether moving
            # after the real small-ball contact is actually feasible.
            self.player_x = paddle_target_left(self.player_x, small_prediction.contact_cells[0], self.config)
            self.ai_x = paddle_target_left(self.ai_x, large_prediction.contact_cells[1], self.config)
            transfer = abs(paddle_target_left(self.player_x, large_prediction.contact_cells[0], self.config) - self.player_x) / self.config.paddle_speed_per_second
            feasible_after_small = small_prediction.time_until_contact + transfer <= large_prediction.time_until_contact + _EPSILON
            self.training_scenario_metadata.update({
                "small_ball_id": small.ball_id, "large_ball_id": large.ball_id,
                "small_contact_time": small_prediction.time_until_contact,
                "large_contact_time": large_prediction.time_until_contact,
                "transfer_seconds_after_small_contact": transfer,
                "small_then_large_feasible": feasible_after_small,
                "small_then_large_variant": "feasible" if feasible_variant else "infeasible",
            })
        elif scenario == "dual_large":
            first = self._active_ball(BallKind.LARGE)
            second = next(ball for ball in self.balls if ball.active and ball.ball_id != first.ball_id)
            first_prediction = self._put_descending_near_contact(first, seconds=self._rng.uniform(.40, .75))
            second_prediction = self._put_descending_near_contact(second, seconds=self._rng.uniform(.75, 1.20))
            self.player_x = paddle_target_left(self.player_x, first_prediction.contact_cells[0], self.config)
            self.ai_x = paddle_target_left(self.ai_x, first_prediction.contact_cells[1], self.config)
            self.training_scenario_metadata.update({"first_ball_id": first.ball_id, "second_ball_id": second.ball_id,
                                                    "first_contact_time": first_prediction.time_until_contact,
                                                    "second_contact_time": second_prediction.time_until_contact})
        elif scenario == "dual_small":
            balls = [ball for ball in self.balls if ball.active]
            first = self._put_descending_near_contact(balls[0], seconds=self._rng.uniform(.40, .85))
            second = self._put_descending_near_contact(balls[1], seconds=self._rng.uniform(.40, .85))
            self.player_x = paddle_target_left(self.player_x, first.contact_cells[0], self.config)
            self.ai_x = paddle_target_left(self.ai_x, second.contact_cells[0], self.config)
            self.training_scenario_metadata.update({"ball_ids": [balls[0].ball_id, balls[1].ball_id]})
        else:
            raise ValueError(f"unknown Pong curriculum scenario {scenario}")

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
                active=True,
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
        own_action, other_action = (
            (self.ai_last_action, self.player_last_action)
            if agent == "ai" else (self.player_last_action, self.ai_last_action)
        )
        action_value = {"left": -1.0, "stay": 0.0, "right": 1.0}
        values: dict[str, float] = {
            "self.paddle_x": own / self.config.width,
            "other.paddle_x": other / self.config.width,
            "time.progress": min(1.0, self.frame_index / max(1, self.config.max_frames)),
            "self.last_action": action_value[own_action],
            "other.last_action": action_value[other_action],
        }
        # A single immutable frame is sufficient for all five predictions.
        # Rebuilding it for every ball used to clone the complete court five
        # times per observation, which made long training and evaluation needlessly slow.
        current_frame = self.frame()
        for ball in self.balls:
            prefix = f"ball.{ball.ball_id}"
            prediction = predict_next_contact(ball, current_frame, self.config)
            current_opportunity = bool(
                ball.active and not ball.descending_encounter and prediction is not None
            )
            # A failed downward encounter is not represented as a still
            # recoverable target while the ball travels to the bottom edge.
            if not current_opportunity:
                prediction = None
            contact_left = prediction.contact_cells[0] if prediction else 0.0
            contact_right = (
                prediction.contact_cells[-1] if prediction and ball.is_large else contact_left
            )
            own_target = paddle_target_left(own, contact_left, self.config) if prediction else own
            other_target = paddle_target_left(other, contact_left, self.config) if prediction else other
            own_distance = abs(own_target - own) if prediction else 0.0
            other_distance = abs(other_target - other) if prediction else 0.0
            time_until = prediction.time_until_contact if prediction else 0.0
            own_reachable = bool(prediction and own_distance / self.config.paddle_speed_per_second <= time_until + _EPSILON)
            other_reachable = bool(prediction and other_distance / self.config.paddle_speed_per_second <= time_until + _EPSILON)
            own_covered = bool(prediction and paddle_covers_cell(own, contact_left, self.config))
            other_covered = bool(prediction and paddle_covers_cell(other, contact_left, self.config))
            self_left_distance = other_right_distance = self_right_distance = other_left_distance = 0.0
            left_assignment = right_assignment = False
            if prediction and ball.is_large:
                left_cell, right_cell = prediction.contact_cells
                self_left_distance = abs(paddle_target_left(own, left_cell, self.config) - own)
                other_right_distance = abs(paddle_target_left(other, right_cell, self.config) - other)
                self_right_distance = abs(paddle_target_left(own, right_cell, self.config) - own)
                other_left_distance = abs(paddle_target_left(other, left_cell, self.config) - other)
                left_assignment = (
                    self_left_distance / self.config.paddle_speed_per_second <= time_until + _EPSILON
                    and other_right_distance / self.config.paddle_speed_per_second <= time_until + _EPSILON
                )
                right_assignment = (
                    self_right_distance / self.config.paddle_speed_per_second <= time_until + _EPSILON
                    and other_left_distance / self.config.paddle_speed_per_second <= time_until + _EPSILON
                )
            values.update({
                f"{prefix}.x": ball.x / self.config.width, f"{prefix}.y": ball.y / self.config.height,
                f"{prefix}.vx": ball.vx / self.config.ball_speed_x_per_second,
                f"{prefix}.vy": ball.vy / self.config.ball_speed_y_per_second,
                f"{prefix}.large": float(ball.is_large), f"{prefix}.descending": float(ball.vy > 0),
                f"{prefix}.encounters": float(ball.encounter_index),
                f"{prefix}.active": float(ball.active),
                f"{prefix}.opportunity_valid": float(current_opportunity),
                f"{prefix}.contact_left": contact_left / self.config.width,
                f"{prefix}.contact_right": contact_right / self.config.width,
                f"{prefix}.time_to_contact": min(1.0, time_until / self.config.duration_seconds),
                f"{prefix}.self_distance": own_distance / self.config.width,
                f"{prefix}.other_distance": other_distance / self.config.width,
                f"{prefix}.self_reachable": float(own_reachable),
                f"{prefix}.other_reachable": float(other_reachable),
                f"{prefix}.self_covers": float(own_covered),
                f"{prefix}.other_covers": float(other_covered),
                f"{prefix}.self_left_distance": self_left_distance / self.config.width,
                f"{prefix}.other_right_distance": other_right_distance / self.config.width,
                f"{prefix}.self_right_distance": self_right_distance / self.config.width,
                f"{prefix}.other_left_distance": other_left_distance / self.config.width,
                f"{prefix}.self_left_assignment_reachable": float(left_assignment),
                f"{prefix}.self_right_assignment_reachable": float(right_assignment),
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
        if not ball.active:
            return []
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
                "encounter_events": deepcopy(self.encounter_events), "history": deepcopy(self.history),
                "training_scenario": self.training_scenario,
                "training_scenario_metadata": deepcopy(getattr(self, "training_scenario_metadata", {}))}

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
        self.training_scenario = str(snapshot.get("training_scenario", "full"))
        self.training_scenario_metadata = deepcopy(snapshot.get("training_scenario_metadata", {}))
        return self.frame()


def _tupleize(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tupleize(item) for item in value)
    return value
